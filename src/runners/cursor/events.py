"""Cursor ACP event helpers."""

from __future__ import annotations

from typing import Any


def extract_session_id(payload: dict[str, Any]) -> str | None:
    params = payload.get("params")
    if not isinstance(params, dict):
        return None
    session_id = params.get("sessionId")
    if session_id is None:
        return None
    return str(session_id)
