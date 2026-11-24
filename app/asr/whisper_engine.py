"""Reusable local OpenAI Whisper transcription engine."""

from __future__ import annotations

import logging
import shutil
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from app.asr.model_manager import ModelManager, supports_fp16
from app.config import SUPPORTED_AUDIO_EXTENSIONS, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)


class AudioValidationError(ValueError):
    """Raised when an audio file is missing, unreadable, or has an unsupported format."""


class TranscriptionError(RuntimeError):
    """Raised when Whisper fails to transcribe an audio file."""


@dataclass(frozen=True)
class TranscriptionResult:
    """Structured output returned by ``WhisperEngine.transcribe``."""

    text: str
    language: str
    language_code: str
    model_name: str
    device: str
    fp16: bool
    processing_time_sec: float
    audio_duration_sec: float
    real_time_factor: float


def map_language_to_code(language: str) -> tuple[str, str]:
    """Map a user-facing language name (or code) to ``(display_name, whisper_code)``."""
    cleaned = language.strip()
    if cleaned in SUPPORTED_LANGUAGES:
        return cleaned, SUPPORTED_LANGUAGES[cleaned]

    lower_map = {name.lower(): (name, code) for name, code in SUPPORTED_LANGUAGES.items()}
    if cleaned.lower() in lower_map:
        return lower_map[cleaned.lower()]

    code_map = {code: (name, code) for name, code in SUPPORTED_LANGUAGES.items()}
    if cleaned.lower() in code_map:
        return code_map[cleaned.lower()]

    supported = ", ".join(SUPPORTED_LANGUAGES.keys())
    raise ValueError(f"Unsupported language '{language}'. Supported languages: {supported}.")


def get_audio_duration(audio_path: Path) -> float:
    """Determine audio clip duration in seconds using SoundFile, Wave, or Librosa.

    Raises:
        AudioValidationError: If the audio file cannot be decoded or has zero duration.
    """
    if not audio_path.exists() or not audio_path.is_file():
        raise AudioValidationError(f"Audio file not found: '{audio_path}'")

    if audio_path.stat().st_size == 0:
        raise AudioValidationError(f"Audio file is empty (0 bytes): '{audio_path}'")

    # 1. Fast metadata read via soundfile (handles WAV, FLAC, OGG, etc.)
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        if info.duration > 0:
            return float(info.duration)
    except Exception:
        pass

    # 2. Built-in wave fallback for standard PCM WAV files
    if audio_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0 and frames > 0:
                    return float(frames / rate)
        except Exception:
            pass

    # 3. Librosa fallback (supports MP3/M4A/FLAC via audioread/ffmpeg)
    try:
        import librosa

        duration = float(librosa.get_duration(path=str(audio_path)))
        if duration > 0:
            return duration
    except Exception as exc:
        raise AudioValidationError(
            f"Unable to read audio file '{audio_path}'. "
            f"Ensure it is a valid audio file and FFmpeg is installed if using MP3/M4A: {exc}"
        ) from exc

    raise AudioValidationError(f"Audio file '{audio_path}' has zero or unreadable duration.")


class WhisperEngine:
    """Reusable local OpenAI Whisper transcription engine supporting Tiny and Base models."""

    def __init__(
        self,
        model_manager: ModelManager | None = None,
        device: str = "auto",
    ) -> None:
        self.model_manager = model_manager or ModelManager()
        self.device = device

    def validate_audio(self, audio_path: str | Path) -> tuple[Path, float]:
        """Validate an audio file path and return ``(resolved_path, duration_sec)``."""
        path = Path(audio_path)
        if not path.exists():
            raise AudioValidationError(f"Audio file does not exist: '{path}'")
        if not path.is_file():
            raise AudioValidationError(f"Audio path is not a file: '{path}'")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_AUDIO_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
            raise AudioValidationError(
                f"Unsupported audio format '{ext}' for '{path.name}'. Supported: {allowed}"
            )

        if ext in {".mp3", ".m4a"} and shutil.which("ffmpeg") is None:
            logger.warning(
                "FFmpeg binary not found on PATH; decoding '%s' may fail.",
                path.name,
            )

        duration_sec = get_audio_duration(path)
        return path, duration_sec

    def transcribe(
        self,
        audio_path: str | Path,
        language: str,
        model_name: str = "tiny",
        device: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe a single audio clip using a locally cached Whisper model.

        Args:
            audio_path: Path to the audio clip (WAV, MP3, M4A, or FLAC).
            language: Target language name (``Hindi``, ``Marathi``, ``English``) or code.
            model_name: Whisper model size (``"tiny"`` or ``"base"``).
            device: Optional device override (``"auto"``, ``"cpu"``, ``"cuda"``).

        Returns:
            ``TranscriptionResult`` with transcript, timing, audio duration, and RTF.
        """
        display_language, language_code = map_language_to_code(language)
        resolved_audio, audio_duration = self.validate_audio(audio_path)

        target_device = device if device is not None else self.device
        model, active_device = self.model_manager.get_model(model_name, device=target_device)
        use_fp16 = supports_fp16(active_device)

        logger.info(
            "Transcribing '%s' | model=%s | language=%s (%s) | device=%s | fp16=%s",
            resolved_audio.name,
            model_name,
            display_language,
            language_code,
            active_device,
            use_fp16,
        )

        start_time = time.perf_counter()
        try:
            raw_result = model.transcribe(
                str(resolved_audio),
                language=language_code,
                task="transcribe",
                fp16=use_fp16,
            )
        except FileNotFoundError as exc:
            raise TranscriptionError(
                f"Transcription failed for '{resolved_audio.name}'. "
                "Verify that FFmpeg is installed and available on your system PATH."
            ) from exc
        except Exception as exc:
            raise TranscriptionError(
                f"Whisper '{model_name}' transcription failed for '{resolved_audio.name}': {exc}"
            ) from exc
        elapsed = time.perf_counter() - start_time

        transcript_text = str(raw_result.get("text", "")).strip()
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0

        return TranscriptionResult(
            text=transcript_text,
            language=display_language,
            language_code=language_code,
            model_name=model_name,
            device=active_device,
            fp16=use_fp16,
            processing_time_sec=round(elapsed, 4),
            audio_duration_sec=round(audio_duration, 4),
            real_time_factor=round(rtf, 4),
        )
