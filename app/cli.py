"""Command-line interface for Multilingual ASR Evaluation."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.config import (
    SUPPORTED_LANGUAGES,
    SUPPORTED_MODELS,
    AppConfig,
    configure_logging,
)
from app.data.manifest_loader import (
    EmptyManifestError,
    ManifestError,
    load_manifest,
)
from app.evaluation.evaluator import ASREvaluator
from app.evaluation.normalizer import NormalizerConfig

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description=(
            "Multilingual ASR Evaluation CLI — evaluate local OpenAI Whisper models "
            "(tiny, base) across Hindi, Marathi, and English audio clips."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: evaluate
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Run Whisper transcription and calculate WER/CER metrics from a manifest CSV.",
    )
    eval_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.csv"),
        help="Path to manifest CSV (default: data/manifest.csv).",
    )
    eval_parser.add_argument(
        "--models",
        nargs="+",
        choices=list(SUPPORTED_MODELS),
        default=list(SUPPORTED_MODELS),
        help="Whisper model size(s) to evaluate (default: tiny base).",
    )
    eval_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device: auto, cpu, or cuda (default: auto).",
    )
    eval_parser.add_argument(
        "--language",
        choices=list(SUPPORTED_LANGUAGES.keys()),
        default=None,
        help="Optional language filter (Hindi, Marathi, or English).",
    )
    eval_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write evaluation_results.csv and evaluation_summary.json.",
    )
    eval_parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable text normalization before computing WER and CER.",
    )
    eval_parser.add_argument(
        "--remove-punctuation",
        action="store_true",
        help="Strip punctuation (including Devanagari Danda) during normalization.",
    )
    eval_parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase text during normalization (affects Latin script only).",
    )

    # Subcommand: validate
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a manifest CSV and referenced audio files without running Whisper.",
    )
    validate_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.csv"),
        help="Path to manifest CSV (default: data/manifest.csv).",
    )
    validate_parser.add_argument(
        "--language",
        choices=list(SUPPORTED_LANGUAGES.keys()),
        default=None,
        help="Optional language filter.",
    )

    return parser


def _format_pct(value: float | None) -> str:
    return f"{value:.2f}%" if value is not None else "N/A"


def _format_num(value: float | None, suffix: str = "") -> str:
    return f"{value:.4f}{suffix}" if value is not None else "N/A"


def _print_summary(summary: dict[str, Any], csv_path: Path, json_path: Path) -> None:
    """Print human-readable evaluation summary, language breakdown, and model comparison."""
    print("\n" + "=" * 72)
    print("MULTILINGUAL ASR EVALUATION SUMMARY")
    print("=" * 72)
    print(f"Manifest           : {summary['manifest_path']}")
    print(f"Device             : {summary['device']}")
    print(f"Models Evaluated   : {', '.join(summary['models_evaluated'])}")
    print(
        f"Normalization      : enabled={summary['normalization']['enabled']} "
        f"(NFC={summary['normalization']['unicode_form']}, "
        f"whitespace={summary['normalization']['normalize_whitespace']}, "
        f"remove_punct={summary['normalization']['remove_punctuation']}, "
        f"lowercase={summary['normalization']['lowercase']})"
    )
    print(
        f"Audio Files        : {summary['total_files']} total | "
        f"{summary['successful_files']} succeeded | "
        f"{summary['partial_files']} partial | "
        f"{summary['failed_files']} failed"
    )

    print("\n--- Overall Model Metrics (Lower WER/CER is better) ---")
    print(f"{'Model':<10} {'Clips':<7} {'WER':<10} {'CER':<10} {'Mean Time':<12} {'Mean RTF':<10}")
    for model_name, stats in summary["overall_metrics"].items():
        print(
            f"{model_name:<10} "
            f"{stats['clips_evaluated']:<7} "
            f"{_format_pct(stats['mean_wer_pct']):<10} "
            f"{_format_pct(stats['mean_cer_pct']):<10} "
            f"{_format_num(stats['mean_processing_time_sec'], 's'):<12} "
            f"{_format_num(stats['mean_rtf']):<10}"
        )

    print("\n--- Language-wise Results ---")
    for lang, lang_info in summary["language_metrics"].items():
        print(f"\nLanguage: {lang} ({lang_info['total_clips']} clip(s))")
        print(f"  {'Model':<10} {'Clips':<7} {'WER':<10} {'CER':<10} {'Mean Time':<12} {'Mean RTF':<10}")
        for model_name, stats in lang_info["models"].items():
            print(
                f"  {model_name:<10} "
                f"{stats['clips_evaluated']:<7} "
                f"{_format_pct(stats['mean_wer_pct']):<10} "
                f"{_format_pct(stats['mean_cer_pct']):<10} "
                f"{_format_num(stats['mean_processing_time_sec'], 's'):<12} "
                f"{_format_num(stats['mean_rtf']):<10}"
            )

    comp = summary.get("model_comparison")
    if comp is not None:
        print("\n--- Tiny vs Base Comparison (Paired Clips) ---")
        print(f"Common Clips       : {comp['common_clips']}")
        print(
            f"Mean WER           : tiny={comp['tiny_mean_wer'] * 100:.2f}% vs "
            f"base={comp['base_mean_wer'] * 100:.2f}% "
            f"(better: {comp['better_wer_model']})"
        )
        print(
            f"Mean CER           : tiny={comp['tiny_mean_cer'] * 100:.2f}% vs "
            f"base={comp['base_mean_cer'] * 100:.2f}% "
            f"(better: {comp['better_cer_model']})"
        )
        print(
            f"Mean Latency       : tiny={comp['tiny_mean_processing_time_sec']:.4f}s vs "
            f"base={comp['base_mean_processing_time_sec']:.4f}s"
        )

    print("\n--- Saved Output Files ---")
    print(f"Detailed CSV : {csv_path}")
    print(f"Summary JSON : {json_path}")
    print("=" * 72 + "\n")


def _handle_validate(args: argparse.Namespace, config: AppConfig) -> int:
    """Validate manifest entries and report counts."""
    try:
        records = load_manifest(
            manifest_path=args.manifest,
            project_root=config.project_root,
            language_filter=args.language,
            allow_empty=False,
        )
    except EmptyManifestError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1
    except ManifestError as exc:
        print(f"Manifest validation error:\n{exc}", file=sys.stderr)
        return 1

    lang_counts: dict[str, int] = {}
    for rec in records:
        lang_counts[rec.language] = lang_counts.get(rec.language, 0) + 1

    breakdown = ", ".join(f"{lang}: {count}" for lang, count in lang_counts.items())
    print(f"Manifest valid: {len(records)} audio clip(s) ready for evaluation ({breakdown}).")
    return 0


def _handle_evaluate(args: argparse.Namespace, config: AppConfig) -> int:
    """Execute the ASR evaluation pipeline."""
    normalizer_config = NormalizerConfig(
        enabled=not args.no_normalize,
        unicode_form="NFC",
        normalize_whitespace=True,
        remove_punctuation=args.remove_punctuation,
        lowercase=args.lowercase,
    )
    evaluator = ASREvaluator(config=config, normalizer_config=normalizer_config)

    def _progress(completed: int, total: int, message: str) -> None:
        print(f"Progress ({completed}/{total}): {message}", flush=True)

    try:
        output = evaluator.evaluate(
            manifest_path=args.manifest,
            models=args.models,
            device=args.device,
            language_filter=args.language,
            output_dir=args.output_dir,
            on_progress=_progress,
        )
    except (ManifestError, ValueError, RuntimeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    _print_summary(output.summary, output.csv_path, output.json_path)

    if output.summary["successful_files"] == 0 and output.summary["partial_files"] == 0:
        print(
            "Evaluation failed: all audio files failed during processing. "
            "Inspect error_message in the CSV output.",
            file=sys.stderr,
        )
        return 1

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning a POSIX exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    config = AppConfig()

    if args.command == "validate":
        return _handle_validate(args, config)
    if args.command == "evaluate":
        return _handle_evaluate(args, config)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
