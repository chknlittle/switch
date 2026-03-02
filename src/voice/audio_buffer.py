"""VAD-based audio segmentation for voice call transcription.

Receives decoded PCM AudioFrames from aiortc (typically 48kHz stereo Opus),
resamples to 16kHz mono (required by webrtcvad and faster-whisper), and
uses voice activity detection to chunk speech into segments suitable for
transcription.

Segments are flushed when:
- Silence exceeds a configurable threshold after speech (~800ms default)
- The buffer hits a maximum duration ceiling (30s default)
"""

from __future__ import annotations

import logging
import os
import struct
import time
from typing import Callable

import numpy as np
import webrtcvad

log = logging.getLogger("voice.audio_buffer")

# webrtcvad operates on 10/20/30ms frames of 16kHz 16-bit mono PCM.
_TARGET_SAMPLE_RATE = 16000
_VAD_FRAME_MS = 30
_VAD_FRAME_SAMPLES = _TARGET_SAMPLE_RATE * _VAD_FRAME_MS // 1000  # 480

# Callback type: receives (pcm_bytes_16khz_mono, sample_rate)
SegmentCallback = Callable[[bytes, int], None]


class AudioBuffer:
    """Accumulates audio, detects speech boundaries, flushes segments."""

    def __init__(self, on_segment: SegmentCallback):
        self._on_segment = on_segment

        # Configuration from env
        try:
            self._silence_ms = int(
                os.getenv("SWITCH_VOICE_VAD_SILENCE_MS", "800")
            )
        except ValueError:
            self._silence_ms = 800

        try:
            self._max_chunk_s = float(
                os.getenv("SWITCH_VOICE_MAX_CHUNK_S", "30")
            )
        except ValueError:
            self._max_chunk_s = 30.0

        # VAD — aggressiveness 2 (0=least aggressive, 3=most aggressive)
        self._vad = webrtcvad.Vad(2)

        # Accumulation state
        self._buffer: list[bytes] = []  # 16kHz mono 16-bit PCM chunks
        self._buffer_samples: int = 0
        self._speech_detected: bool = False
        self._last_speech_time: float = 0.0
        self._vad_accum: bytes = b""  # partial frame accumulator

    def push_frame(self, frame) -> None:
        """Push an aiortc AudioFrame into the buffer.

        Resamples to 16kHz mono and runs VAD on each 30ms chunk.
        """
        try:
            pcm_16k = self._resample_frame(frame)
        except Exception:
            log.debug("Failed to resample audio frame", exc_info=True)
            return

        # Prepend any leftover from previous frame
        data = self._vad_accum + pcm_16k
        self._vad_accum = b""

        frame_bytes = _VAD_FRAME_SAMPLES * 2  # 16-bit = 2 bytes per sample
        offset = 0

        while offset + frame_bytes <= len(data):
            chunk = data[offset : offset + frame_bytes]
            offset += frame_bytes

            is_speech = False
            try:
                is_speech = self._vad.is_speech(chunk, _TARGET_SAMPLE_RATE)
            except Exception:
                pass

            now = time.monotonic()

            if is_speech:
                self._speech_detected = True
                self._last_speech_time = now

            # Always accumulate when speech has been detected
            if self._speech_detected:
                self._buffer.append(chunk)
                self._buffer_samples += _VAD_FRAME_SAMPLES

            # Check flush conditions
            buffer_duration = self._buffer_samples / _TARGET_SAMPLE_RATE

            if self._speech_detected and not is_speech:
                silence_duration_ms = (now - self._last_speech_time) * 1000
                if silence_duration_ms >= self._silence_ms:
                    self._flush()
                    continue

            if buffer_duration >= self._max_chunk_s:
                self._flush()

        # Save leftover partial frame
        if offset < len(data):
            self._vad_accum = data[offset:]

    def flush_remaining(self) -> None:
        """Flush any remaining buffered audio (e.g. on call end)."""
        if self._buffer and self._speech_detected:
            self._flush()

    def _flush(self) -> None:
        """Flush accumulated speech buffer to the segment callback."""
        if not self._buffer:
            self._reset()
            return

        pcm = b"".join(self._buffer)
        samples = self._buffer_samples
        duration_s = samples / _TARGET_SAMPLE_RATE

        self._reset()

        # Skip very short segments (likely noise)
        if duration_s < 0.3:
            log.debug("Skipping short segment: %.2fs", duration_s)
            return

        log.info("Flushing audio segment: %.2fs (%d samples)", duration_s, samples)
        try:
            self._on_segment(pcm, _TARGET_SAMPLE_RATE)
        except Exception:
            log.exception("Segment callback failed")

    def _reset(self) -> None:
        """Reset accumulation state for next segment."""
        self._buffer.clear()
        self._buffer_samples = 0
        self._speech_detected = False
        self._last_speech_time = 0.0

    @staticmethod
    def _resample_frame(frame) -> bytes:
        """Resample an aiortc AudioFrame to 16kHz mono 16-bit PCM.

        aiortc typically delivers 48kHz stereo (2ch) Opus-decoded frames.
        """
        # frame.to_ndarray() returns shape (samples, channels) as int16
        arr = frame.to_ndarray()

        # Mono mix if stereo
        if arr.ndim == 2 and arr.shape[1] > 1:
            arr = arr.mean(axis=1).astype(np.int16)
        elif arr.ndim == 2:
            arr = arr[:, 0]

        src_rate = frame.sample_rate
        if src_rate != _TARGET_SAMPLE_RATE:
            # Simple linear interpolation resampling
            src_len = len(arr)
            dst_len = int(src_len * _TARGET_SAMPLE_RATE / src_rate)
            if dst_len == 0:
                return b""
            indices = np.linspace(0, src_len - 1, dst_len)
            arr = np.interp(indices, np.arange(src_len), arr.astype(np.float64))
            arr = arr.astype(np.int16)

        return arr.tobytes()
