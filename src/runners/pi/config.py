"""Pi runner configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PiConfig:
    model: str | None = None
    provider: str | None = None
    thinking: str | None = None
    session_dir: str | None = None

    # Optional overrides (otherwise env defaults apply)
    pi_bin: str | None = None

    # System prompt to append via --append-system-prompt.
    # None = use default, "" = skip entirely.
    system_prompt: str | None = None

    def resolve_bin(self) -> str:
        return self.pi_bin or os.getenv("PI_BIN", "pi")

    def resolve_provider(self) -> str | None:
        return self.provider or os.getenv("PI_PROVIDER") or None

    def resolve_model(self) -> str | None:
        return self.model or os.getenv("PI_MODEL") or None

    def resolve_session_dir(self) -> str | None:
        return self.session_dir or os.getenv("PI_SESSION_DIR") or None
