"""Unit tests for manifest CSV loading and validation."""

from __future__ import annotations

import wave
from pathlib import Path

import pandas as pd
import pytest

from app.data.manifest_loader import (
    EmptyManifestError,
    ManifestNotFoundError,
    ManifestValidationError,
    load_manifest,
)


def _create_dummy_wav(path: Path) -> None:
    """Create a tiny valid PCM WAV file inside a temporary test directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)


def _write_manifest(path: Path, rows: list[dict[str, str]], columns: list[str] | None = None) -> None:
    cols = columns or ["audio_path", "language", "reference_text"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(path, index=False, encoding="utf-8")


def test_valid_manifest_and_relative_path_resolution(tmp_path: Path) -> None:
    hi_wav = tmp_path / "data" / "audio" / "hindi_1.wav"
    mr_wav = tmp_path / "data" / "audio" / "marathi_1.wav"
    en_wav = tmp_path / "data" / "audio" / "english_1.wav"
    en_wav_2 = tmp_path / "data" / "audio" / "english_2.wav"

    for wav_file in (hi_wav, mr_wav, en_wav, en_wav_2):
        _create_dummy_wav(wav_file)

    manifest_file = tmp_path / "data" / "manifest.csv"
    _write_manifest(
        manifest_file,
        [
            {
                "audio_path": "data/audio/hindi_1.wav",
                "language": "Hindi",
                "reference_text": "नमस्ते भारत",
            },
            {
                "audio_path": "data/audio/marathi_1.wav",
                "language": "Marathi",
                "reference_text": "नमस्कार महाराष्ट्र",
            },
            {
                "audio_path": "data/audio/english_1.wav",
                "language": "English",
                "reference_text": "hello world",
            },
            {
                "audio_path": "data/audio/english_2.wav",
                "language": "English",
                "reference_text": "second english clip",
            },
        ],
    )

    records = load_manifest("data/manifest.csv", project_root=tmp_path)
    assert len(records) == 4
    assert records[0].resolved_audio_path == hi_wav.resolve()
    assert records[0].language_code == "hi"
    assert records[1].language_code == "mr"
    assert records[2].language_code == "en"

    english_only = load_manifest(manifest_file, project_root=tmp_path, language_filter="English")
    assert len(english_only) == 2


def test_missing_required_columns(tmp_path: Path) -> None:
    manifest_file = tmp_path / "manifest.csv"
    _write_manifest(
        manifest_file,
        [{"audio_path": "data/audio/a.wav", "language": "English"}],
        columns=["audio_path", "language"],
    )

    with pytest.raises(ManifestValidationError, match="missing required column"):
        load_manifest(manifest_file, project_root=tmp_path)


def test_invalid_language_rejected(tmp_path: Path) -> None:
    wav_file = tmp_path / "data" / "audio" / "clip.wav"
    _create_dummy_wav(wav_file)
    manifest_file = tmp_path / "manifest.csv"
    _write_manifest(
        manifest_file,
        [
            {
                "audio_path": "data/audio/clip.wav",
                "language": "French",
                "reference_text": "bonjour",
            }
        ],
    )

    with pytest.raises(ManifestValidationError, match="unsupported language 'French'"):
        load_manifest(manifest_file, project_root=tmp_path)


def test_empty_reference_transcript_rejected(tmp_path: Path) -> None:
    wav_file = tmp_path / "data" / "audio" / "clip.wav"
    _create_dummy_wav(wav_file)
    manifest_file = tmp_path / "manifest.csv"
    _write_manifest(
        manifest_file,
        [
            {
                "audio_path": "data/audio/clip.wav",
                "language": "Hindi",
                "reference_text": "   ",
            }
        ],
    )

    with pytest.raises(ManifestValidationError, match="reference_text is empty"):
        load_manifest(manifest_file, project_root=tmp_path)


def test_missing_audio_file_rejected(tmp_path: Path) -> None:
    manifest_file = tmp_path / "manifest.csv"
    _write_manifest(
        manifest_file,
        [
            {
                "audio_path": "data/audio/does_not_exist.wav",
                "language": "Marathi",
                "reference_text": "चाचणी वाक्य",
            }
        ],
    )

    with pytest.raises(ManifestValidationError, match="audio file does not exist"):
        load_manifest(manifest_file, project_root=tmp_path)


def test_empty_manifest_handling(tmp_path: Path) -> None:
    manifest_file = tmp_path / "manifest.csv"
    _write_manifest(manifest_file, [])

    with pytest.raises(EmptyManifestError, match="contains no audio records"):
        load_manifest(manifest_file, project_root=tmp_path)

    records = load_manifest(manifest_file, project_root=tmp_path, allow_empty=True)
    assert records == []


def test_missing_manifest_file_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(ManifestNotFoundError):
        load_manifest("missing.csv", project_root=tmp_path)
