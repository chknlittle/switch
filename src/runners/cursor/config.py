"""Cursor ACP runner configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CursorModelVariant:
    """User-configurable settings encoded in a Cursor model ID."""

    family: str
    thinking: str
    fast: bool


# Keep variant knowledge at the Cursor adapter boundary. Commands remain
# model-neutral and can support another family by adding its levels here.
_MODEL_VARIANT_LEVELS: dict[str, frozenset[str]] = {
    "cursor-grok-4.5": frozenset({"low", "medium", "high"}),
    "cursor-grok-4.6": frozenset({"low", "medium", "high", "xhigh"}),
    "grok-4.7": frozenset({"low", "medium", "high", "xhigh"}),
}


def parse_model_variant(model: str | None) -> CursorModelVariant | None:
    """Decode a supported Cursor model ID into user-facing settings."""
    model_id = (model or "").strip().lower()
    fast = model_id.endswith("-fast")
    without_speed = model_id[:-5] if fast else model_id
    for family, levels in _MODEL_VARIANT_LEVELS.items():
        prefix = f"{family}-"
        if without_speed.startswith(prefix):
            thinking = without_speed[len(prefix) :]
            if thinking in levels:
                return CursorModelVariant(family, thinking, fast)
    return None


def resolve_model_variant(
    model: str | None,
    *,
    thinking: str | None = None,
    fast: bool | None = None,
) -> str | None:
    """Return a model ID with selected settings, preserving unspecified ones."""
    variant = parse_model_variant(model)
    if variant is None:
        return None
    selected_thinking = (thinking or variant.thinking).strip().lower()
    if selected_thinking not in _MODEL_VARIANT_LEVELS[variant.family]:
        return None
    selected_fast = variant.fast if fast is None else fast
    suffix = "-fast" if selected_fast else ""
    return f"{variant.family}-{selected_thinking}{suffix}"


@dataclass(frozen=True)
class CursorConfig:
    model: str | None = None
    cursor_bin: str | None = None
    auth_method: str = "cursor_login"
    permission_option_id: str = "allow-once"
    control_plane_timeout_s: float | None = None
    post_message_idle_timeout_s: float | None = None

    def resolve_bin(self) -> str:
        return self.cursor_bin or os.getenv("CURSOR_AGENT_BIN", "agent")

    def resolve_model(self) -> str:
        return self.model or os.getenv("CURSOR_MODEL", "composer-2.5")
