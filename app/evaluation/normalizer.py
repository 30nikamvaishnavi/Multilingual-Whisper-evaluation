"""Configurable multilingual text normalization preserving Devanagari scripts."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

UnicodeForm = Literal["NFC", "NFKC", "NFD", "NFKD"]


@dataclass(frozen=True)
class NormalizerConfig:
    """Configuration controlling text normalization prior to WER/CER scoring."""

    enabled: bool = True
    unicode_form: UnicodeForm | None = "NFC"
    normalize_whitespace: bool = True
    remove_punctuation: bool = False
    lowercase: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration for evaluation reports and reproducibility logs."""
        return {
            "enabled": self.enabled,
            "unicode_form": self.unicode_form if self.enabled else None,
            "normalize_whitespace": self.normalize_whitespace if self.enabled else False,
            "remove_punctuation": self.remove_punctuation if self.enabled else False,
            "lowercase": self.lowercase if self.enabled else False,
        }


def _strip_punctuation_preserve_script(text: str, replace_with_space: bool) -> str:
    """Remove Unicode punctuation (category ``P*``) while keeping letters, digits, and combining marks.

    Because Devanagari vowel signs (matras), virama, anusvara, visarga, chandrabindu, and nukta
    belong to Unicode letter/mark categories (``Lo``, ``Mn``, ``Mc``), while Hindi/Marathi
    Danda (``।``) and Double Danda (``॥``) belong to punctuation (``Po``), checking Unicode
    category ``P*`` safely strips punctuation without altering meaningful Devanagari characters.
    """
    replacement = " " if replace_with_space else ""
    chars: list[str] = []
    for ch in text:
        if unicodedata.category(ch).startswith("P"):
            chars.append(replacement)
        else:
            chars.append(ch)
    return "".join(chars)


class TextNormalizer:
    """Configurable text normalizer supporting Hindi, Marathi, and English transcripts."""

    def __init__(self, config: NormalizerConfig | None = None) -> None:
        self.config = config or NormalizerConfig()

    def normalize(self, text: str) -> str:
        """Return a normalized copy of ``text`` without mutating the original string.

        Args:
            text: Input transcript text.

        Returns:
            Normalized string according to ``self.config``.

        Raises:
            TypeError: If ``text`` is not a ``str``.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be str, got {type(text).__name__}.")

        if not self.config.enabled:
            return text

        result = text

        if self.config.unicode_form is not None:
            result = unicodedata.normalize(self.config.unicode_form, result)

        if self.config.remove_punctuation:
            result = _strip_punctuation_preserve_script(
                result,
                replace_with_space=self.config.normalize_whitespace,
            )

        if self.config.lowercase:
            result = result.lower()

        if self.config.normalize_whitespace:
            result = re.sub(r"\s+", " ", result).strip()

        return result

    def describe(self) -> dict[str, Any]:
        """Return a dictionary describing the active normalization configuration."""
        return self.config.to_dict()


def normalize_text(text: str, config: NormalizerConfig | None = None) -> str:
    """Functional helper to normalize a string using ``NormalizerConfig``."""
    return TextNormalizer(config=config).normalize(text)
