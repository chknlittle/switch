"""OpenCode runner configuration.

Configuration lives at the adapter boundary so higher-level code doesn't grow a
dependency on OpenCodeRunner's internal constructor signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.runners.opencode.models import QuestionCallback


@dataclass(frozen=True)
class OpenCodeConfig:
    model: str | None = None
    reasoning_mode: str = "normal"
    agent: str = "bridge"
    server_url: str | None = None
    question_callback: QuestionCallback | None = None
    permission_reply: Literal["once", "always", "reject"] = "always"
    permission_message: str | None = None

    # Optional overrides (otherwise env defaults apply)
    http_timeout_s: float | None = None
    post_message_idle_timeout_s: float | None = None
