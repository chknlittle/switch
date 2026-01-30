"""Ports for SessionRuntime.

These interfaces keep the runtime independent of transport (XMPP), storage
(SQLite), and runner implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.runners import Runner, RunnerEvent
from src.runners.opencode.config import OpenCodeConfig
from src.attachments import Attachment


@dataclass(frozen=True)
class SessionState:
    name: str
    active_engine: str
    claude_session_id: str | None
    opencode_session_id: str | None
    opencode_agent: str
    model_id: str
    reasoning_mode: str


class SessionStorePort(Protocol):
    def get(self, name: str) -> SessionState | None: ...

    def update_last_active(self, name: str) -> None: ...

    def update_claude_session_id(self, name: str, session_id: str) -> None: ...

    def update_opencode_session_id(self, name: str, session_id: str) -> None: ...


class MessageStorePort(Protocol):
    def add(self, session_name: str, role: str, content: str, engine: str) -> None: ...


class RunnerFactoryPort(Protocol):
    def create(
        self,
        engine: str,
        *,
        working_dir: str,
        output_dir: Path,
        session_name: str,
        opencode_config: OpenCodeConfig | None = None,
    ) -> Runner: ...


class HistoryPort(Protocol):
    def append_to_history(self, message: str, working_dir: str, claude_session_id: str | None) -> None: ...

    def log_activity(self, message: str, *, session: str, source: str) -> None: ...


class RunnerEventSinkPort(Protocol):
    async def on_event(self, event: RunnerEvent) -> None: ...


class AttachmentPromptPort(Protocol):
    def augment_prompt(self, body: str, attachments: list[Attachment] | None) -> str: ...


class RalphLoopStorePort(Protocol):
    def create(
        self,
        session_name: str,
        prompt: str,
        max_iterations: int,
        completion_promise: str | None,
        wait_seconds: float,
    ) -> int: ...

    def update_progress(
        self,
        loop_id: int,
        current_iteration: int,
        total_cost: float,
        status: str = "running",
    ) -> None: ...
