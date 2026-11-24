"""Unit tests for WER and CER calculation in app.evaluation.metrics."""

from __future__ import annotations

import pytest

from app.evaluation.metrics import calculate_cer, calculate_wer, compute_text_metrics
from app.evaluation.normalizer import NormalizerConfig, TextNormalizer


def test_identical_reference_and_hypothesis() -> None:
    assert calculate_wer("hello world", "hello world") == pytest.approx(0.0)
    assert calculate_cer("hello world", "hello world") == pytest.approx(0.0)


def test_one_substitution_wer() -> None:
    # 1 substitution out of 4 words -> 0.25
    wer = calculate_wer("the quick brown fox", "the fast brown fox")
    assert wer == pytest.approx(0.25)


def test_one_deletion_wer() -> None:
    # 1 deletion out of 4 words -> 0.25
    wer = calculate_wer("the quick brown fox", "the quick fox")
    assert wer == pytest.approx(0.25)


def test_one_insertion_wer() -> None:
    # 1 insertion out of 4 reference words -> 0.25
    wer = calculate_wer("the quick brown fox", "the very quick brown fox")
    assert wer == pytest.approx(0.25)


def test_empty_reference_handling() -> None:
    assert calculate_wer("", "") == pytest.approx(0.0)
    assert calculate_cer("", "") == pytest.approx(0.0)
    assert calculate_wer("", "spoken text") == pytest.approx(1.0)
    assert calculate_cer("", "spoken text") == pytest.approx(1.0)
    assert calculate_wer("   ", "spoken text") == pytest.approx(1.0)


def test_empty_hypothesis_handling() -> None:
    assert calculate_wer("reference sentence here", "") == pytest.approx(1.0)
    assert calculate_cer("reference sentence here", "") == pytest.approx(1.0)
    assert calculate_wer("reference sentence", "   ") == pytest.approx(1.0)


def test_hindi_unicode_text_wer_and_cer() -> None:
    ref = "आज मौसम बहुत सुहावना है"
    hyp_exact = "आज मौसम बहुत सुहावना है"
    hyp_one_sub = "आज मौसम काफी सुहावना है"

    assert calculate_wer(ref, hyp_exact) == pytest.approx(0.0)
    assert calculate_cer(ref, hyp_exact) == pytest.approx(0.0)
    # 1 word substituted out of 5 words -> 0.20
    assert calculate_wer(ref, hyp_one_sub) == pytest.approx(0.20)
    assert 0.0 < calculate_cer(ref, hyp_one_sub) < 1.0


def test_marathi_unicode_text_wer_and_cer() -> None:
    ref = "मला मराठी भाषा खूप आवडते"
    hyp_exact = "मला मराठी भाषा खूप आवडते"
    hyp_missing_word = "मला मराठी खूप आवडते"

    assert calculate_wer(ref, hyp_exact) == pytest.approx(0.0)
    assert calculate_cer(ref, hyp_exact) == pytest.approx(0.0)
    # 1 deletion out of 5 words -> 0.20
    assert calculate_wer(ref, hyp_missing_word) == pytest.approx(0.20)
    assert 0.0 < calculate_cer(ref, hyp_missing_word) < 1.0


def test_english_text_wer_and_percentages() -> None:
    result = compute_text_metrics(
        "speech recognition evaluation pipeline",
        "speech recognition evaluation system",
    )
    assert result.wer == pytest.approx(0.25)
    assert result.wer_percent == pytest.approx(25.0)
    assert result.cer > 0.0
    assert result.cer_percent == pytest.approx(result.cer * 100.0, abs=0.01)
    assert result.normalization_enabled is False


def test_cer_calculation_exact_characters() -> None:
    # "cat" -> "bat": 1 substitution out of 3 chars = 1/3
    assert calculate_cer("cat", "bat") == pytest.approx(1.0 / 3.0)
    # "राम" (3 chars: र, ा, म) -> "शाम" (3 chars: श, ा, म): 1 substitution out of 3 chars
    assert calculate_cer("राम", "शाम") == pytest.approx(1.0 / 3.0)


def test_metrics_input_validation_and_normalization_flag() -> None:
    with pytest.raises(TypeError):
        calculate_wer(None, "hello")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        calculate_cer("hello", 123)  # type: ignore[arg-type]

    normalizer = TextNormalizer(NormalizerConfig(enabled=True, remove_punctuation=True))
    result = compute_text_metrics("Hello, world!", "Hello world", normalizer=normalizer)
    assert result.wer == pytest.approx(0.0)
    assert result.cer == pytest.approx(0.0)
    assert result.normalization_enabled is True
