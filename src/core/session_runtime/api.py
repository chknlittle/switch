"""Public API for session runtime.

This module is the stable boundary between:
- transport/adapters (XMPP, commands)
- the concrete runtime implementation (runtime.py)

Code outside the runtime should depend on these types/protocols, not on
SessionRuntime internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.attachments import Attachment


@dataclass(frozen=True)
class RalphConfig:
    prompt: str
    max_iterations: int = 0
    completion_promise: str | None = None
    wait_seconds: float = 2.0
    force_engine: str = "claude"


@dataclass
class RalphStatus:
    status: str  # queued|running|stopping|completed|cancelled|error|max_iterations|finished
    current_iteration: int = 0
    max_iterations: int = 0
    wait_seconds: float = 0.0
    completion_promise: str | None = None
    total_cost: float = 0.0
    loop_id: int | None = None
    error: str | None = None


class SessionPort(Protocol):
    def pending_count(self) -> int: ...

    async def enqueue(
        self,
        body: str,
        attachments: list[Attachment] | None,
        *,
        trigger_response: bool,
        scheduled: bool,
        wait: bool,
    ) -> None: ...

    def cancel_operations(self, *, notify: bool = False) -> bool: ...

    def shutdown(self) -> None: ...

    def answer_question(self, answer: object, *, request_id: str | None = None) -> bool: ...

    async def start_ralph(self, cfg: RalphConfig, *, wait: bool = False) -> None: ...

    def request_ralph_stop(self) -> bool: ...

    def get_ralph_status(self) -> RalphStatus | None: ...


# -----------------
# Event boundary
# -----------------


@dataclass(frozen=True)
class OutboundMessage:
    text: str
    meta_type: str | None = None
    meta_tool: str | None = None
    meta_attrs: dict[str, str] | None = None
    meta_payload: object | None = None


@dataclass(frozen=True)
class ProcessingChanged:
    active: bool


SessionEvent = OutboundMessage | ProcessingChanged


class EventSinkPort(Protocol):
    async def emit(self, event: SessionEvent) -> None: ...
