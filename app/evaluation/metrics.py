"""Word Error Rate (WER) and Character Error Rate (CER) evaluation using jiwer."""

from __future__ import annotations

from dataclasses import dataclass

import jiwer

from app.evaluation.normalizer import TextNormalizer


@dataclass(frozen=True)
class MetricResult:
    """Structured WER and CER evaluation result for a reference/hypothesis pair."""

    wer: float
    cer: float
    wer_percent: float
    cer_percent: float
    normalization_enabled: bool


def _validate_inputs(reference: str, hypothesis: str) -> None:
    """Validate that both reference and hypothesis are strings."""
    if not isinstance(reference, str):
        raise TypeError(f"reference must be a str, got {type(reference).__name__}.")
    if not isinstance(hypothesis, str):
        raise TypeError(f"hypothesis must be a str, got {type(hypothesis).__name__}.")


def _prepare_texts(
    reference: str,
    hypothesis: str,
    normalizer: TextNormalizer | None = None,
) -> tuple[str, str]:
    """Validate inputs and optionally apply non-mutating normalization."""
    _validate_inputs(reference, hypothesis)
    if normalizer is not None:
        return normalizer.normalize(reference), normalizer.normalize(hypothesis)
    return reference, hypothesis


def calculate_wer(
    reference: str,
    hypothesis: str,
    normalizer: TextNormalizer | None = None,
) -> float:
    """Calculate Word Error Rate (WER) between ``reference`` and ``hypothesis``.

    Uses ``jiwer.wer`` with whitespace word tokenization so Unicode Devanagari and Latin
    scripts are evaluated faithfully without implicit English-only transformations.

    Empty input handling:
    - If both reference and hypothesis are empty (or whitespace-only), returns ``0.0``.
    - If reference is empty while hypothesis is non-empty, returns ``1.0``.
    - If reference is non-empty while hypothesis is empty, returns ``1.0``.

    Returns:
        WER as a non-negative floating-point rate (e.g., ``0.25`` equals ``25.0%``).
    """
    ref_text, hyp_text = _prepare_texts(reference, hypothesis, normalizer=normalizer)
    ref_stripped = ref_text.strip()
    hyp_stripped = hyp_text.strip()

    if not ref_stripped and not hyp_stripped:
        return 0.0
    if not ref_stripped:
        return 1.0
    if not hyp_stripped:
        return 1.0

    transform = jiwer.Compose([jiwer.Strip(), jiwer.RemoveMultipleSpaces(), jiwer.ReduceToListOfListOfWords()])
    score = jiwer.wer(
        ref_stripped,
        hyp_stripped,
        reference_transform=transform,
        hypothesis_transform=transform,
    )
    return float(score)


def calculate_cer(
    reference: str,
    hypothesis: str,
    normalizer: TextNormalizer | None = None,
) -> float:
    """Calculate Character Error Rate (CER) between ``reference`` and ``hypothesis``.

    Uses ``jiwer.cer`` directly on Unicode characters so Devanagari and Latin characters
    are preserved and compared accurately.

    Empty input handling:
    - If both reference and hypothesis are empty (or whitespace-only), returns ``0.0``.
    - If reference is empty while hypothesis is non-empty, returns ``1.0``.
    - If reference is non-empty while hypothesis is empty, returns ``1.0``.

    Returns:
        CER as a non-negative floating-point rate (e.g., ``0.10`` equals ``10.0%``).
    """
    ref_text, hyp_text = _prepare_texts(reference, hypothesis, normalizer=normalizer)
    ref_stripped = ref_text.strip()
    hyp_stripped = hyp_text.strip()

    if not ref_stripped and not hyp_stripped:
        return 0.0
    if not ref_stripped:
        return 1.0
    if not hyp_stripped:
        return 1.0

    transform = jiwer.Compose([jiwer.Strip(), jiwer.RemoveMultipleSpaces(), jiwer.ReduceToListOfListOfChars()])
    score = jiwer.cer(
        ref_stripped,
        hyp_stripped,
        reference_transform=transform,
        hypothesis_transform=transform,
    )
    return float(score)


def compute_text_metrics(
    reference: str,
    hypothesis: str,
    normalizer: TextNormalizer | None = None,
) -> MetricResult:
    """Compute WER and CER rates and percentages for a transcript pair."""
    wer_rate = calculate_wer(reference, hypothesis, normalizer=normalizer)
    cer_rate = calculate_cer(reference, hypothesis, normalizer=normalizer)
    return MetricResult(
        wer=round(wer_rate, 6),
        cer=round(cer_rate, 6),
        wer_percent=round(wer_rate * 100.0, 2),
        cer_percent=round(cer_rate * 100.0, 2),
        normalization_enabled=bool(normalizer is not None and normalizer.config.enabled),
    )
