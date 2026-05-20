"""Cursor ACP runner configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CursorConfig:
    model: str | None = None
    cursor_bin: str | None = None
    auth_method: str = "cursor_login"
    permission_option_id: str = "allow-once"
    request_timeout_s: float = 600.0

    def resolve_bin(self) -> str:
        return self.cursor_bin or os.getenv("CURSOR_AGENT_BIN", "agent")

    def resolve_model(self) -> str:
        return self.model or os.getenv("CURSOR_MODEL", "composer-2.5")
