"""Manifest CSV loading, validation, and dataset management utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.config import (
    MANIFEST_REQUIRED_COLUMNS,
    PROJECT_ROOT,
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_LANGUAGES,
)

logger = logging.getLogger(__name__)


class ManifestError(Exception):
    """Base exception for manifest-related errors."""


class ManifestNotFoundError(ManifestError, FileNotFoundError):
    """Raised when the manifest CSV file cannot be found."""


class ManifestValidationError(ManifestError, ValueError):
    """Raised when the manifest schema or one or more rows are invalid."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors: list[str] = errors or [message]


class EmptyManifestError(ManifestValidationError):
    """Raised when the manifest contains no records."""


@dataclass(frozen=True)
class ManifestRecord:
    """A single validated audio clip entry from the evaluation manifest."""

    audio_path: str
    resolved_audio_path: Path
    language: str
    language_code: str
    reference_text: str
    row_number: int

    @property
    def audio_filename(self) -> str:
        """Return the base filename of the audio clip."""
        return self.resolved_audio_path.name


def resolve_audio_path(raw_path: str, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve an audio path relative to ``project_root`` unless already absolute."""
    path = Path(raw_path.strip())
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def validate_manifest_dataframe(
    df: pd.DataFrame,
    project_root: Path = PROJECT_ROOT,
    supported_languages: dict[str, str] = SUPPORTED_LANGUAGES,
    supported_extensions: frozenset[str] = SUPPORTED_AUDIO_EXTENSIONS,
) -> tuple[list[ManifestRecord], list[str]]:
    """Validate a manifest DataFrame and return valid records plus row-level error messages.

    Args:
        df: Loaded manifest DataFrame.
        project_root: Base directory used to resolve relative ``audio_path`` entries.
        supported_languages: Mapping of valid display language names to Whisper codes.
        supported_extensions: Set of lowercase supported audio file extensions.

    Returns:
        A tuple of ``(valid_records, error_messages)``.

    Raises:
        ManifestValidationError: If required columns are missing from the DataFrame.
    """
    normalized_cols = [str(col).strip() for col in df.columns]
    missing_columns = [col for col in MANIFEST_REQUIRED_COLUMNS if col not in normalized_cols]
    if missing_columns:
        raise ManifestValidationError(
            f"Manifest is missing required column(s): {', '.join(missing_columns)}. "
            f"Expected columns: {', '.join(MANIFEST_REQUIRED_COLUMNS)}."
        )

    df = df.copy()
    df.columns = normalized_cols

    records: list[ManifestRecord] = []
    errors: list[str] = []

    for idx, row in df.iterrows():
        csv_row_num = int(idx) + 2  # Account for 1-based indexing and CSV header row
        raw_audio_path = str(row.get("audio_path", "")).strip()
        language = str(row.get("language", "")).strip()
        reference_text = str(row.get("reference_text", "")).strip()

        row_errors: list[str] = []

        if not raw_audio_path:
            row_errors.append("audio_path is empty")
            resolved_path = Path()
        else:
            resolved_path = resolve_audio_path(raw_audio_path, project_root=project_root)
            if not resolved_path.exists():
                row_errors.append(
                    f"audio file does not exist: '{raw_audio_path}' (resolved to '{resolved_path}')"
                )
            elif not resolved_path.is_file():
                row_errors.append(f"audio_path is not a file: '{resolved_path}'")
            elif resolved_path.suffix.lower() not in supported_extensions:
                allowed = ", ".join(sorted(supported_extensions))
                row_errors.append(
                    f"unsupported audio format '{resolved_path.suffix}' "
                    f"(supported: {allowed})"
                )

        if not language:
            row_errors.append("language is empty")
        elif language not in supported_languages:
            allowed_langs = ", ".join(supported_languages.keys())
            row_errors.append(
                f"unsupported language '{language}' (supported: {allowed_langs})"
            )

        if not reference_text:
            row_errors.append("reference_text is empty")

        if row_errors:
            errors.append(f"Row {csv_row_num}: {'; '.join(row_errors)}")
            continue

        records.append(
            ManifestRecord(
                audio_path=raw_audio_path,
                resolved_audio_path=resolved_path,
                language=language,
                language_code=supported_languages[language],
                reference_text=reference_text,
                row_number=csv_row_num,
            )
        )

    return records, errors


def load_manifest(
    manifest_path: str | Path,
    project_root: Path = PROJECT_ROOT,
    language_filter: str | None = None,
    allow_empty: bool = False,
) -> list[ManifestRecord]:
    """Load and validate an evaluation manifest CSV file.

    Args:
        manifest_path: Path to the manifest CSV (relative to ``project_root`` or absolute).
        project_root: Project root used to resolve relative paths.
        language_filter: Optional language name (``Hindi``, ``Marathi``, ``English``) to filter.
        allow_empty: If ``True``, return an empty list when the manifest has zero rows instead
            of raising ``EmptyManifestError``.

    Returns:
        List of validated ``ManifestRecord`` objects.

    Raises:
        ManifestNotFoundError: If ``manifest_path`` does not exist.
        EmptyManifestError: If the manifest has zero records (when ``allow_empty=False``) or
            zero records match ``language_filter``.
        ManifestValidationError: If schema or row validation fails.
    """
    path = Path(manifest_path)
    if not path.is_absolute():
        path = (project_root / path).resolve()

    if not path.exists():
        raise ManifestNotFoundError(f"Manifest file not found: '{path}'")
    if not path.is_file():
        raise ManifestValidationError(f"Manifest path is not a file: '{path}'")

    if language_filter is not None and language_filter not in SUPPORTED_LANGUAGES:
        allowed_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
        raise ManifestValidationError(
            f"Unsupported language filter '{language_filter}'. Supported: {allowed_langs}."
        )

    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except pd.errors.EmptyDataError as exc:
        raise ManifestValidationError(
            f"Manifest file '{path}' is completely empty and missing the CSV header."
        ) from exc
    except Exception as exc:
        raise ManifestValidationError(f"Failed to read manifest CSV '{path}': {exc}") from exc

    records, errors = validate_manifest_dataframe(df, project_root=project_root)

    if errors:
        joined = "\n  - ".join(errors)
        raise ManifestValidationError(
            f"Manifest validation failed with {len(errors)} error(s):\n  - {joined}",
            errors=errors,
        )

    if not records:
        if allow_empty:
            logger.info("Manifest '%s' contains header only (0 records).", path)
            return []
        raise EmptyManifestError(
            f"Manifest '{path}' contains no audio records. "
            "Add audio clips to data/audio/ and rows to data/manifest.csv before evaluating."
        )

    if language_filter is not None:
        filtered = [record for record in records if record.language == language_filter]
        if not filtered:
            if allow_empty:
                return []
            raise EmptyManifestError(
                f"No records found in manifest '{path}' for language '{language_filter}'."
            )
        return filtered

    return records


def append_manifest_record(
    manifest_path: str | Path,
    audio_path: str,
    language: str,
    reference_text: str,
    project_root: Path = PROJECT_ROOT,
) -> ManifestRecord:
    """Validate and append a single audio record to the manifest CSV.

    Args:
        manifest_path: Path to the target manifest CSV.
        audio_path: Audio file path (preferably relative to ``project_root``).
        language: One of ``Hindi``, ``Marathi``, or ``English``.
        reference_text: Verified reference transcript in the target script.
        project_root: Project root used to resolve relative paths.

    Returns:
        The newly appended and validated ``ManifestRecord``.
    """
    path = Path(manifest_path)
    if not path.is_absolute():
        path = (project_root / path).resolve()

    if path.exists():
        existing_df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
        missing = [c for c in MANIFEST_REQUIRED_COLUMNS if c not in existing_df.columns]
        if missing:
            raise ManifestValidationError(
                f"Existing manifest is missing required column(s): {', '.join(missing)}"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_df = pd.DataFrame(columns=list(MANIFEST_REQUIRED_COLUMNS))

    new_row_df = pd.DataFrame(
        [
            {
                "audio_path": audio_path.strip(),
                "language": language.strip(),
                "reference_text": reference_text.strip(),
            }
        ]
    )
    records, errors = validate_manifest_dataframe(new_row_df, project_root=project_root)
    if errors or not records:
        raise ManifestValidationError(
            f"Cannot add invalid record: {'; '.join(errors)}",
            errors=errors,
        )

    updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    updated_df.to_csv(path, index=False, encoding="utf-8")
    return records[0]


def save_manifest_records(
    manifest_path: str | Path,
    df: pd.DataFrame,
    project_root: Path = PROJECT_ROOT,
) -> list[ManifestRecord]:
    """Validate an uploaded or edited manifest DataFrame and save it to disk."""
    path = Path(manifest_path)
    if not path.is_absolute():
        path = (project_root / path).resolve()

    records, errors = validate_manifest_dataframe(df, project_root=project_root)
    if errors:
        joined = "\n  - ".join(errors)
        raise ManifestValidationError(
            f"Manifest validation failed with {len(errors)} error(s):\n  - {joined}",
            errors=errors,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    df.loc[:, list(MANIFEST_REQUIRED_COLUMNS)].to_csv(path, index=False, encoding="utf-8")
    return records
