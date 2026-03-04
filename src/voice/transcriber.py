"""Async wrapper around faster-whisper for speech-to-text.

Loads the WhisperModel once on first use and runs transcription in a
ThreadPoolExecutor to avoid blocking the asyncio event loop.

No subprocess involved → no stderr pipe deadlock concern (unlike the
Pi/Claude subprocess runners — see MEMORY.md).
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

log = logging.getLogger("voice.transcriber")

# Singleton model instance — loaded lazily on first transcribe() call.
_model = None
_model_lock = asyncio.Lock()

# Thread pool for blocking whisper inference.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")


def _get_model_config() -> tuple[str, str]:
    """Read model size and device from environment."""
    model_size = os.getenv("SWITCH_WHISPER_MODEL", "base").strip() or "base"
    device = os.getenv("SWITCH_WHISPER_DEVICE", "auto").strip() or "auto"
    return model_size, device


def _load_model():
    """Load the faster-whisper model (blocking — run in executor)."""
    from faster_whisper import WhisperModel

    model_size, device = _get_model_config()

    if device == "auto":
        # faster-whisper auto-selects CUDA if available
        compute_type = "float16" if _cuda_available() else "int8"
        actual_device = "cuda" if _cuda_available() else "cpu"
    elif device == "cuda":
        compute_type = "float16"
        actual_device = "cuda"
    else:
        compute_type = "int8"
        actual_device = "cpu"

    log.info(
        "Loading faster-whisper model=%s device=%s compute_type=%s",
        model_size,
        actual_device,
        compute_type,
    )
    return WhisperModel(model_size, device=actual_device, compute_type=compute_type)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _transcribe_sync(pcm: bytes, sample_rate: int) -> str:
    """Run whisper transcription synchronously (called from executor thread)."""
    global _model

    if _model is None:
        _model = _load_model()

    # Convert raw PCM bytes to float32 numpy array normalized to [-1, 1]
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    segments, info = _model.transcribe(
        audio,
        beam_size=5,
        language=None,  # auto-detect
        vad_filter=True,
    )

    # Collect all segment texts
    texts = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            texts.append(text)

    result = " ".join(texts).strip()
    if result:
        log.info(
            "Transcribed %.1fs audio (lang=%s, prob=%.2f): %s",
            info.duration,
            info.language,
            info.language_probability,
            result[:100],
        )
    else:
        log.debug("No speech detected in %.1fs audio", info.duration)

    return result


async def transcribe(pcm: bytes, sample_rate: int) -> str:
    """Transcribe PCM audio to text asynchronously.

    Args:
        pcm: Raw 16-bit signed integer PCM bytes
        sample_rate: Sample rate of the PCM data (typically 16000)

    Returns:
        Transcribed text, or empty string if no speech detected.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _transcribe_sync, pcm, sample_rate)


async def ensure_model_loaded() -> None:
    """Pre-load the whisper model (call during startup to avoid first-call latency)."""
    async with _model_lock:
        global _model
        if _model is not None:
            return
        loop = asyncio.get_running_loop()
        _model = await loop.run_in_executor(_executor, _load_model)
