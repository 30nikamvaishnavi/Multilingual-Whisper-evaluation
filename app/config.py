"""Configuration and constants for the Multilingual ASR Evaluation platform."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_LANGUAGES: dict[str, str] = {
    "Hindi": "hi",
    "Marathi": "mr",
    "English": "en",
}

SUPPORTED_MODELS: tuple[str, ...] = ("tiny", "base")

SUPPORTED_AUDIO_EXTENSIONS: frozenset[str] = frozenset({".wav", ".mp3", ".m4a", ".flac"})

MANIFEST_REQUIRED_COLUMNS: tuple[str, ...] = ("audio_path", "language", "reference_text")

DEFAULT_RESULTS_CSV_NAME = "evaluation_results.csv"
DEFAULT_SUMMARY_JSON_NAME = "evaluation_summary.json"


@dataclass(frozen=True)
class AppConfig:
    """Centralized, immutable configuration for paths, models, and languages."""

    project_root: Path = PROJECT_ROOT
    manifest_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "manifest.csv")
    audio_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "audio")
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")
    model_cache_dir: Path | None = field(
        default_factory=lambda: (
            Path(os.environ["WHISPER_CACHE_DIR"]).expanduser()
            if os.environ.get("WHISPER_CACHE_DIR")
            else None
        )
    )
    supported_languages: dict[str, str] = field(default_factory=lambda: dict(SUPPORTED_LANGUAGES))
    supported_models: tuple[str, ...] = SUPPORTED_MODELS
    supported_audio_extensions: frozenset[str] = SUPPORTED_AUDIO_EXTENSIONS

    def resolve_path(self, path_value: str | Path) -> Path:
        """Resolve a path relative to ``project_root`` if it is not already absolute."""
        candidate = Path(path_value)
        if candidate.is_absolute():
            return candidate
        return (self.project_root / candidate).resolve()


def configure_logging(level: int = logging.INFO) -> None:
    """Configure clean application-wide logging."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        root_logger.setLevel(level)
