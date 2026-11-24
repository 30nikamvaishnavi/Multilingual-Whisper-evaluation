"""End-to-end multilingual ASR evaluation pipeline."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.asr.model_manager import ModelManager, resolve_device
from app.asr.whisper_engine import WhisperEngine, get_audio_duration
from app.config import (
    DEFAULT_RESULTS_CSV_NAME,
    DEFAULT_SUMMARY_JSON_NAME,
    PROJECT_ROOT,
    SUPPORTED_MODELS,
    AppConfig,
)
from app.data.manifest_loader import ManifestRecord, load_manifest
from app.evaluation.metrics import compute_text_metrics
from app.evaluation.normalizer import NormalizerConfig, TextNormalizer

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]

RESULT_COLUMNS: tuple[str, ...] = (
    "audio_filename",
    "audio_path",
    "language",
    "reference_transcript",
    "tiny_transcription",
    "base_transcription",
    "tiny_wer",
    "base_wer",
    "tiny_wer_pct",
    "base_wer_pct",
    "tiny_cer",
    "base_cer",
    "tiny_cer_pct",
    "base_cer_pct",
    "tiny_processing_time_sec",
    "base_processing_time_sec",
    "audio_duration_sec",
    "tiny_rtf",
    "base_rtf",
    "status",
    "error_message",
)


class EvaluationError(RuntimeError):
    """Raised when evaluation cannot be executed."""


@dataclass
class EvaluationOutput:
    """Container returned after executing an evaluation run."""

    results_df: pd.DataFrame
    summary: dict[str, Any]
    csv_path: Path
    json_path: Path


def _safe_mean(series: pd.Series) -> float | None:
    """Return rounded mean of non-null numeric values, or None if empty."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return round(float(clean.mean()), 6)


def _safe_sum(series: pd.Series) -> float | None:
    """Return rounded sum of non-null numeric values, or None if empty."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return round(float(clean.sum()), 4)


def _compute_model_stats(df: pd.DataFrame, model_name: str) -> dict[str, Any]:
    """Compute aggregate statistics for one Whisper model from detailed result rows."""
    wer_col = f"{model_name}_wer"
    cer_col = f"{model_name}_cer"
    time_col = f"{model_name}_processing_time_sec"
    rtf_col = f"{model_name}_rtf"

    valid_mask = pd.to_numeric(df[wer_col], errors="coerce").notna()
    valid_df = df[valid_mask]
    count = int(len(valid_df))
    if count == 0:
        return {
            "clips_evaluated": 0,
            "mean_wer": None,
            "mean_wer_pct": None,
            "mean_cer": None,
            "mean_cer_pct": None,
            "mean_processing_time_sec": None,
            "total_processing_time_sec": None,
            "mean_rtf": None,
            "total_audio_duration_sec": None,
        }

    mean_wer = _safe_mean(valid_df[wer_col])
    mean_cer = _safe_mean(valid_df[cer_col])

    return {
        "clips_evaluated": count,
        "mean_wer": mean_wer,
        "mean_wer_pct": round(mean_wer * 100.0, 2) if mean_wer is not None else None,
        "mean_cer": mean_cer,
        "mean_cer_pct": round(mean_cer * 100.0, 2) if mean_cer is not None else None,
        "mean_processing_time_sec": _safe_mean(valid_df[time_col]),
        "total_processing_time_sec": _safe_sum(valid_df[time_col]),
        "mean_rtf": _safe_mean(valid_df[rtf_col]),
        "total_audio_duration_sec": _safe_sum(valid_df["audio_duration_sec"]),
    }


def _compute_paired_comparison(df: pd.DataFrame) -> dict[str, Any] | None:
    """Compare Tiny vs Base on the exact same successfully evaluated clips."""
    both_valid = df[
        pd.to_numeric(df["tiny_wer"], errors="coerce").notna()
        & pd.to_numeric(df["base_wer"], errors="coerce").notna()
    ]
    if both_valid.empty:
        return None

    tiny_wer = float(pd.to_numeric(both_valid["tiny_wer"]).mean())
    base_wer = float(pd.to_numeric(both_valid["base_wer"]).mean())
    tiny_cer = float(pd.to_numeric(both_valid["tiny_cer"]).mean())
    base_cer = float(pd.to_numeric(both_valid["base_cer"]).mean())
    tiny_time = float(pd.to_numeric(both_valid["tiny_processing_time_sec"]).mean())
    base_time = float(pd.to_numeric(both_valid["base_processing_time_sec"]).mean())

    def _pick_better(tiny_val: float, base_val: float) -> str:
        if abs(tiny_val - base_val) < 1e-9:
            return "tie"
        return "base" if base_val < tiny_val else "tiny"

    return {
        "common_clips": int(len(both_valid)),
        "tiny_mean_wer": round(tiny_wer, 6),
        "base_mean_wer": round(base_wer, 6),
        "wer_delta_base_minus_tiny": round(base_wer - tiny_wer, 6),
        "tiny_mean_cer": round(tiny_cer, 6),
        "base_mean_cer": round(base_cer, 6),
        "cer_delta_base_minus_tiny": round(base_cer - tiny_cer, 6),
        "tiny_mean_processing_time_sec": round(tiny_time, 4),
        "base_mean_processing_time_sec": round(base_time, 4),
        "better_wer_model": _pick_better(tiny_wer, base_wer),
        "better_cer_model": _pick_better(tiny_cer, base_cer),
    }


class ASREvaluator:
    """Runs Whisper Tiny and Base evaluation over a validated audio manifest."""

    def __init__(
        self,
        config: AppConfig | None = None,
        normalizer_config: NormalizerConfig | None = None,
        engine: WhisperEngine | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self.normalizer = TextNormalizer(config=normalizer_config or NormalizerConfig())
        self.engine = engine or WhisperEngine(
            model_manager=ModelManager(download_root=self.config.model_cache_dir)
        )

    def evaluate(
        self,
        manifest_path: str | Path | None = None,
        models: tuple[str, ...] | list[str] = SUPPORTED_MODELS,
        device: str = "auto",
        language_filter: str | None = None,
        output_dir: str | Path | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> EvaluationOutput:
        """Execute end-to-end evaluation and save CSV + JSON reports.

        Args:
            manifest_path: Path to manifest CSV (defaults to ``config.manifest_path``).
            models: Whisper models to run (subset/order of ``("tiny", "base")``).
            device: Execution device (``"auto"``, ``"cpu"``, ``"cuda"``).
            language_filter: Optional language name (``Hindi``, ``Marathi``, ``English``).
            output_dir: Directory where CSV and JSON outputs are saved.
            on_progress: Optional callback receiving ``(completed_steps, total_steps, message)``.

        Returns:
            ``EvaluationOutput`` containing DataFrame, summary dictionary, and output file paths.
        """
        selected_models = tuple(dict.fromkeys(m.strip().lower() for m in models))
        if not selected_models:
            raise ValueError("At least one Whisper model ('tiny' or 'base') must be selected.")
        invalid_models = [m for m in selected_models if m not in SUPPORTED_MODELS]
        if invalid_models:
            raise ValueError(
                f"Unsupported model(s): {', '.join(invalid_models)}. "
                f"Supported: {', '.join(SUPPORTED_MODELS)}."
            )

        resolved_device = resolve_device(device)
        target_manifest = Path(manifest_path) if manifest_path is not None else self.config.manifest_path
        if not target_manifest.is_absolute():
            target_manifest = (self.config.project_root / target_manifest).resolve()

        records = load_manifest(
            manifest_path=target_manifest,
            project_root=self.config.project_root,
            language_filter=language_filter,
            allow_empty=False,
        )

        target_output_dir = Path(output_dir) if output_dir is not None else self.config.results_dir
        if not target_output_dir.is_absolute():
            target_output_dir = (self.config.project_root / target_output_dir).resolve()
        target_output_dir.mkdir(parents=True, exist_ok=True)

        total_steps = len(records) * len(selected_models)
        completed_steps = 0
        rows: list[dict[str, Any]] = []

        for record in records:
            row = self._evaluate_single_record(
                record=record,
                models=selected_models,
                device=resolved_device,
                on_step_done=lambda msg: None,
            )
            # Emit progress per model step cleanly
            for model_name in selected_models:
                completed_steps += 1
                if on_progress is not None:
                    on_progress(
                        completed_steps,
                        total_steps,
                        f"[{completed_steps}/{total_steps}] {record.audio_filename} "
                        f"({record.language}) - Whisper {model_name}",
                    )
            rows.append(row)

        results_df = pd.DataFrame(rows, columns=list(RESULT_COLUMNS))
        summary = self._build_summary(
            df=results_df,
            manifest_path=target_manifest,
            models=selected_models,
            device=resolved_device,
            language_filter=language_filter,
        )

        csv_path = target_output_dir / DEFAULT_RESULTS_CSV_NAME
        json_path = target_output_dir / DEFAULT_SUMMARY_JSON_NAME

        results_df.to_csv(csv_path, index=False, encoding="utf-8")
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)

        logger.info("Saved detailed evaluation results to '%s'.", csv_path)
        logger.info("Saved evaluation summary to '%s'.", json_path)

        return EvaluationOutput(
            results_df=results_df,
            summary=summary,
            csv_path=csv_path,
            json_path=json_path,
        )

    def _evaluate_single_record(
        self,
        record: ManifestRecord,
        models: tuple[str, ...],
        device: str,
        on_step_done: Callable[[str], None],
    ) -> dict[str, Any]:
        """Evaluate one manifest entry across all selected Whisper models."""
        row: dict[str, Any] = {
            "audio_filename": record.audio_filename,
            "audio_path": record.audio_path,
            "language": record.language,
            "reference_transcript": record.reference_text,
            "tiny_transcription": None,
            "base_transcription": None,
            "tiny_wer": None,
            "base_wer": None,
            "tiny_wer_pct": None,
            "base_wer_pct": None,
            "tiny_cer": None,
            "base_cer": None,
            "tiny_cer_pct": None,
            "base_cer_pct": None,
            "tiny_processing_time_sec": None,
            "base_processing_time_sec": None,
            "audio_duration_sec": None,
            "tiny_rtf": None,
            "base_rtf": None,
            "status": "pending",
            "error_message": "",
        }

        try:
            row["audio_duration_sec"] = round(get_audio_duration(record.resolved_audio_path), 4)
        except Exception as exc:
            logger.error("Audio validation failed for '%s': %s", record.audio_filename, exc)
            row["status"] = "failed"
            row["error_message"] = str(exc)
            return row

        errors: list[str] = []
        succeeded_models = 0

        for model_name in models:
            try:
                transcription = self.engine.transcribe(
                    audio_path=record.resolved_audio_path,
                    language=record.language,
                    model_name=model_name,
                    device=device,
                )
                metrics = compute_text_metrics(
                    reference=record.reference_text,
                    hypothesis=transcription.text,
                    normalizer=self.normalizer,
                )
                row[f"{model_name}_transcription"] = transcription.text
                row[f"{model_name}_wer"] = metrics.wer
                row[f"{model_name}_wer_pct"] = metrics.wer_percent
                row[f"{model_name}_cer"] = metrics.cer
                row[f"{model_name}_cer_pct"] = metrics.cer_percent
                row[f"{model_name}_processing_time_sec"] = transcription.processing_time_sec
                row[f"{model_name}_rtf"] = transcription.real_time_factor
                row["audio_duration_sec"] = transcription.audio_duration_sec
                succeeded_models += 1
                on_step_done(model_name)
            except Exception as exc:
                logger.error(
                    "Model '%s' failed on '%s': %s",
                    model_name,
                    record.audio_filename,
                    exc,
                )
                errors.append(f"{model_name}: {exc}")

        if succeeded_models == len(models):
            row["status"] = "success"
        elif succeeded_models > 0:
            row["status"] = "partial"
        else:
            row["status"] = "failed"

        row["error_message"] = " | ".join(errors)
        return row

    def _build_summary(
        self,
        df: pd.DataFrame,
        manifest_path: Path,
        models: tuple[str, ...],
        device: str,
        language_filter: str | None,
    ) -> dict[str, Any]:
        """Build a reproducible JSON-serializable summary of evaluation metrics."""
        try:
            rel_manifest = str(manifest_path.relative_to(PROJECT_ROOT))
        except ValueError:
            rel_manifest = str(manifest_path)

        total_files = int(len(df))
        successful_files = int((df["status"] == "success").sum())
        partial_files = int((df["status"] == "partial").sum())
        failed_files = int((df["status"] == "failed").sum())

        overall_metrics: dict[str, Any] = {}
        for model_name in models:
            overall_metrics[model_name] = _compute_model_stats(df, model_name)

        language_metrics: dict[str, Any] = {}
        for language_name, lang_df in df.groupby("language", sort=False):
            language_metrics[str(language_name)] = {
                "total_clips": int(len(lang_df)),
                "models": {
                    model_name: _compute_model_stats(lang_df, model_name)
                    for model_name in models
                },
            }

        model_comparison = (
            _compute_paired_comparison(df)
            if "tiny" in models and "base" in models
            else None
        )

        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_path": rel_manifest,
            "device": device,
            "models_evaluated": list(models),
            "language_filter": language_filter,
            "normalization": self.normalizer.describe(),
            "total_files": total_files,
            "successful_files": successful_files,
            "partial_files": partial_files,
            "failed_files": failed_files,
            "overall_metrics": overall_metrics,
            "language_metrics": language_metrics,
            "model_comparison": model_comparison,
        }
