"""Claude runner configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClaudeConfig:
    model: str | None = None
