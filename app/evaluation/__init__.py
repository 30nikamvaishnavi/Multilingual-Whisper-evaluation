"""Evaluation metrics, Unicode text normalization, and batch evaluator."""

from app.evaluation.evaluator import ASREvaluator, EvaluationError, EvaluationOutput
from app.evaluation.metrics import MetricResult, calculate_cer, calculate_wer, compute_text_metrics
from app.evaluation.normalizer import NormalizerConfig, TextNormalizer, normalize_text

__all__ = [
    "ASREvaluator",
    "EvaluationError",
    "EvaluationOutput",
    "MetricResult",
    "NormalizerConfig",
    "TextNormalizer",
    "calculate_cer",
    "calculate_wer",
    "compute_text_metrics",
    "normalize_text",
]
