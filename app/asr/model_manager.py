"""Whisper model loading, device selection, and in-memory caching."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import SUPPORTED_MODELS

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when a Whisper model cannot be loaded or required dependencies are missing."""


def is_cuda_available() -> bool:
    """Return True if PyTorch CUDA is available."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def resolve_device(requested_device: str = "auto") -> str:
    """Resolve execution device (``cpu`` or ``cuda``) from user preference.

    Args:
        requested_device: One of ``"auto"``, ``"cpu"``, or ``"cuda"`` (or ``"cuda:N"``).

    Returns:
        Resolved device string (``"cpu"`` or ``"cuda"``/``"cuda:N"``).

    Raises:
        ValueError: If ``requested_device`` is empty or invalid.
        ModelLoadError: If CUDA is explicitly requested when unavailable or PyTorch is missing.
    """
    normalized = requested_device.strip().lower()
    if normalized not in {"auto", "cpu"} and not normalized.startswith("cuda"):
        raise ValueError(
            f"Unsupported device '{requested_device}'. Expected 'auto', 'cpu', or 'cuda'."
        )

    try:
        import torch
    except ImportError as exc:
        raise ModelLoadError(
            "PyTorch is required to run Whisper models. Install dependencies via requirements.txt."
        ) from exc

    cuda_ready = bool(torch.cuda.is_available())

    if normalized == "auto":
        return "cuda" if cuda_ready else "cpu"

    if normalized.startswith("cuda") and not cuda_ready:
        raise ModelLoadError(
            "CUDA device was explicitly requested, but no CUDA-capable GPU is available."
        )

    return normalized


def supports_fp16(device: str) -> bool:
    """Return True only when FP16 inference is supported on the selected device."""
    return device.strip().lower().startswith("cuda")


class ModelManager:
    """Manages local downloading and in-memory caching of OpenAI Whisper models."""

    def __init__(self, download_root: Path | None = None) -> None:
        self._download_root = download_root
        self._cache: dict[tuple[str, str], Any] = {}

    @property
    def download_root(self) -> Path:
        """Return the local cache directory where Whisper weights are stored."""
        if self._download_root is not None:
            return self._download_root
        return Path.home() / ".cache" / "whisper"

    def is_cached(self, model_name: str, device: str) -> bool:
        """Check whether ``(model_name, device)`` is currently loaded in memory."""
        return (model_name, device) in self._cache

    def clear_cache(self) -> None:
        """Release all cached model instances from memory."""
        self._cache.clear()

    def get_model(self, model_name: str, device: str = "auto") -> tuple[Any, str]:
        """Load and cache a local Whisper model.

        Args:
            model_name: Whisper model size (``"tiny"`` or ``"base"``).
            device: Target execution device (``"auto"``, ``"cpu"``, or ``"cuda"``).

        Returns:
            Tuple of ``(loaded_whisper_model, resolved_device)``.

        Raises:
            ValueError: If ``model_name`` is not supported.
            ModelLoadError: If ``openai-whisper`` is missing or loading fails.
        """
        normalized_name = model_name.strip().lower()
        if normalized_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported Whisper model '{model_name}'. "
                f"Supported models: {', '.join(SUPPORTED_MODELS)}."
            )

        resolved_device = resolve_device(device)
        cache_key = (normalized_name, resolved_device)

        if cache_key in self._cache:
            logger.debug(
                "Using cached Whisper model '%s' on device '%s'.",
                normalized_name,
                resolved_device,
            )
            return self._cache[cache_key], resolved_device

        try:
            import whisper
        except ImportError as exc:
            raise ModelLoadError(
                "The 'openai-whisper' package is not installed. "
                "Install it with `pip install openai-whisper`."
            ) from exc

        download_root_str = str(self._download_root) if self._download_root is not None else None
        logger.info(
            "Loading Whisper model '%s' on device '%s' (cache: %s)...",
            normalized_name,
            resolved_device,
            self.download_root,
        )

        try:
            model = whisper.load_model(
                normalized_name,
                device=resolved_device,
                download_root=download_root_str,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load Whisper model '{normalized_name}' on '{resolved_device}': {exc}"
            ) from exc

        self._cache[cache_key] = model
        return model, resolved_device
