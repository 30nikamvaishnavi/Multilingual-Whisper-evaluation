"""Unit tests for Unicode and Devanagari-safe text normalization."""

from __future__ import annotations

import unicodedata

import pytest

from app.evaluation.normalizer import NormalizerConfig, TextNormalizer, normalize_text


def test_unicode_nfc_normalization() -> None:
    # Decomposed NFD form of "café" and Devanagari nukta character "क़" (क + ़)
    decomposed_latin = unicodedata.normalize("NFD", "café résumé")
    composed_latin = unicodedata.normalize("NFC", "café résumé")
    assert decomposed_latin != composed_latin

    normalizer = TextNormalizer(NormalizerConfig(unicode_form="NFC"))
    assert normalizer.normalize(decomposed_latin) == composed_latin

    decomposed_devanagari = "\u0915\u093c"  # क + nukta
    normalized_devanagari = normalizer.normalize(decomposed_devanagari)
    assert normalized_devanagari == unicodedata.normalize("NFC", decomposed_devanagari)


def test_whitespace_normalization() -> None:
    raw = "  आज   मौसम \t बहुत\nसुहावना   है  "
    normalizer = TextNormalizer(NormalizerConfig(normalize_whitespace=True))
    assert normalizer.normalize(raw) == "आज मौसम बहुत सुहावना है"

    preserve_ws = TextNormalizer(NormalizerConfig(normalize_whitespace=False))
    assert preserve_ws.normalize("  a   b ") == "  a   b "


def test_devanagari_preservation_with_and_without_punctuation() -> None:
    hindi_text = "नमस्ते, भारत! मैं ठीक हूँ।"
    marathi_text = "मी पुण्याहून बोलत आहे; धन्यवाद॥"

    # Default config keeps punctuation and all Devanagari characters intact
    default_norm = TextNormalizer()
    assert default_norm.normalize(hindi_text) == hindi_text
    assert default_norm.normalize(marathi_text) == marathi_text

    # With punctuation removal enabled, matras/chandrabindu/anusvara/virama remain intact
    punct_norm = TextNormalizer(NormalizerConfig(remove_punctuation=True))
    assert punct_norm.normalize(hindi_text) == "नमस्ते भारत मैं ठीक हूँ"
    assert punct_norm.normalize(marathi_text) == "मी पुण्याहून बोलत आहे धन्यवाद"


def test_punctuation_configuration_and_lowercase() -> None:
    english = "Hello, World! Is ASR working?"

    keep_punct = TextNormalizer(NormalizerConfig(remove_punctuation=False, lowercase=False))
    assert keep_punct.normalize(english) == "Hello, World! Is ASR working?"

    strip_and_lower = TextNormalizer(NormalizerConfig(remove_punctuation=True, lowercase=True))
    assert strip_and_lower.normalize(english) == "hello world is asr working"


def test_disabled_normalization_and_describe() -> None:
    raw = "  Mixed   Spacing, Test!  "
    cfg = NormalizerConfig(enabled=False, remove_punctuation=True, lowercase=True)
    normalizer = TextNormalizer(cfg)

    assert normalizer.normalize(raw) == raw
    assert normalizer.describe() == {
        "enabled": False,
        "unicode_form": None,
        "normalize_whitespace": False,
        "remove_punctuation": False,
        "lowercase": False,
    }


def test_normalize_text_type_validation() -> None:
    with pytest.raises(TypeError):
        normalize_text(12345)  # type: ignore[arg-type]
