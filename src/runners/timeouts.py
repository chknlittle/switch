"""Shared timeout policy for streaming agent runners.

Control-plane RPCs (handshakes, session open) get short wall-clock limits.
Agent turns stream events for an unbounded duration; only stall after the
request completes and trailing events go quiet.
"""

from __future__ import annotations

import os


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return float(str(raw).strip())


def control_plane_timeout_s(*, override: float | None = None) -> float:
    """Short timeout for setup RPCs (initialize, session/new, auth)."""
    if override is not None:
        return override
    return _env_float("SWITCH_CONTROL_PLANE_TIMEOUT_S", 120.0)


def post_message_idle_timeout_s(*, override: float | None = None) -> float:
    """After the prompt/message RPC completes, wait this long for trailing events."""
    if override is not None:
        return override
    if os.getenv("SWITCH_POST_MESSAGE_IDLE_TIMEOUT_S") is not None:
        return _env_float("SWITCH_POST_MESSAGE_IDLE_TIMEOUT_S", 30.0)
    # Backward compatibility with the OpenCode-specific env var.
    return _env_float("OPENCODE_POST_MESSAGE_IDLE_TIMEOUT_S", 30.0)


def runner_idle_stall_timeout_s(*, override: float | None = None) -> float:
    """Line/stream readers warn (but keep waiting) after this much silence."""
    if override is not None:
        return override
    return _env_float("SWITCH_RUNNER_IDLE_STALL_TIMEOUT_S", 300.0)
