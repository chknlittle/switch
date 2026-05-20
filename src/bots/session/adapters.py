"""Port adapters wiring SessionBot / DB repos into SessionRuntime ports."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from src.attachments import Attachment
from src.core.session_runtime.api import (
    EventSinkPort,
    OutboundMessage,
    ProcessingChanged,
    SessionEvent,
)
from src.core.session_runtime.ports import (
    AttachmentPromptPort,
    HistoryPort,
    MessageStorePort,
    RalphLoopStorePort,
    RunnerFactoryPort,
    SessionState,
    SessionStorePort,
)
from src.db import MessageRepository, RalphLoopRepository, SessionRepository
from src.helpers import append_to_history, log_activity
from src.runners import Runner, create_runner
from src.runners.claude.config import ClaudeConfig
from src.runners.cursor.config import CursorConfig
from src.runners.opencode.config import OpenCodeConfig
from src.runners.pi.config import PiConfig

if TYPE_CHECKING:
    from src.bots.session.typing import TypingIndicator


class _EventSinkBot(Protocol):
    processing: bool
    _typing: TypingIndicator

    def send_reply(
        self,
        text: str,
        recipient: str | None = None,
        *,
        meta_type: str | None = None,
        meta_tool: str | None = None,
        meta_attrs: dict[str, str] | None = None,
        meta_payload: object | None = None,
    ) -> None: ...


class EventSinkAdapter(EventSinkPort):
    def __init__(self, bot: _EventSinkBot):
        self._bot = bot

    async def emit(self, event: SessionEvent) -> None:
        if isinstance(event, ProcessingChanged):
            self._bot.processing = event.active
            if event.active:
                self._bot._typing.start()
            else:
                self._bot._typing.stop()
            return

        if isinstance(event, OutboundMessage):
            self._bot.send_reply(
                event.text,
                meta_type=event.meta_type,
                meta_tool=event.meta_tool,
                meta_attrs=event.meta_attrs,
                meta_payload=event.meta_payload,
            )
            return


class SessionsAdapter(SessionStorePort):
    def __init__(self, repo: SessionRepository):
        self._repo = repo

    def get(self, name: str) -> SessionState | None:
        s = self._repo.get(name)
        if not s:
            return None
        return SessionState(
            name=s.name,
            active_engine=s.active_engine,
            claude_session_id=s.claude_session_id,
            opencode_session_id=s.opencode_session_id,
            pi_session_id=s.pi_session_id,
            cursor_session_id=s.cursor_session_id,
            model_id=s.model_id,
            reasoning_mode=s.reasoning_mode,
            opencode_agent=s.opencode_agent,
        )

    async def update_last_active(self, name: str) -> None:
        await self._repo.update_last_active(name)

    async def update_claude_session_id(self, name: str, session_id: str) -> None:
        await self._repo.update_claude_session_id(name, session_id)

    async def update_pi_session_id(self, name: str, session_id: str) -> None:
        await self._repo.update_pi_session_id(name, session_id)

    async def update_opencode_session_id(self, name: str, session_id: str) -> None:
        await self._repo.update_opencode_session_id(name, session_id)

    async def update_cursor_session_id(self, name: str, session_id: str) -> None:
        await self._repo.update_cursor_session_id(name, session_id)

    async def update_remote_session_id(
        self, name: str, engine: str, session_id: str
    ) -> None:
        await self._repo.update_remote_session_id(name, engine, session_id)


class MessagesAdapter(MessageStorePort):
    def __init__(self, repo: MessageRepository):
        self._repo = repo

    async def add(
        self, session_name: str, role: str, content: str, engine: str
    ) -> None:
        await self._repo.add(session_name, role, content, engine)


class RunnerFactoryAdapter(RunnerFactoryPort):
    def create(
        self,
        engine: str,
        *,
        working_dir: str,
        output_dir: Path,
        session_name: str,
        pi_config: PiConfig | None = None,
        opencode_config: OpenCodeConfig | None = None,
        claude_config: ClaudeConfig | None = None,
        cursor_config: CursorConfig | None = None,
    ) -> Runner:
        return create_runner(
            engine,
            working_dir=working_dir,
            output_dir=output_dir,
            session_name=session_name,
            pi_config=pi_config,
            opencode_config=opencode_config,
            claude_config=claude_config,
            cursor_config=cursor_config,
        )


class HistoryAdapter(HistoryPort):
    def append_to_history(
        self, message: str, working_dir: str, claude_session_id: str | None
    ) -> None:
        append_to_history(message, working_dir, claude_session_id)

    def log_activity(self, message: str, *, session: str, source: str) -> None:
        log_activity(message, session=session, source=source)


class PromptAdapter(AttachmentPromptPort):
    def augment_prompt(
        self, body: str, attachments: list[Attachment] | None
    ) -> str:
        if not attachments:
            return (body or "").strip()
        lines: list[str] = [(body or "").strip(), "", "User attached image(s):"]
        for a in attachments:
            lines.append(f"- {a.local_path}")
        return "\n".join(lines).strip()


class RalphLoopsAdapter(RalphLoopStorePort):
    def __init__(self, repo: RalphLoopRepository):
        self._repo = repo

    async def create(
        self,
        session_name: str,
        prompt: str,
        max_iterations: int,
        completion_promise: str | None,
        wait_seconds: float,
    ) -> int:
        return await self._repo.create(
            session_name,
            prompt,
            max_iterations=max_iterations,
            completion_promise=completion_promise,
            wait_seconds=float(wait_seconds or 0.0),
        )

    async def update_progress(
        self,
        loop_id: int,
        current_iteration: int,
        total_cost: float,
        status: str = "running",
    ) -> None:
        await self._repo.update_progress(
            loop_id, current_iteration, total_cost, status
        )


def build_runtime_adapters(
    bot: _EventSinkBot,
    *,
    sessions: SessionRepository,
    messages: MessageRepository,
    ralph_loops: RalphLoopRepository,
) -> dict[str, Any]:
    """Construct port adapters for SessionRuntime wiring."""
    return {
        "sessions": SessionsAdapter(sessions),
        "messages": MessagesAdapter(messages),
        "events": EventSinkAdapter(bot),
        "runner_factory": RunnerFactoryAdapter(),
        "history": HistoryAdapter(),
        "prompt": PromptAdapter(),
        "ralph_loops": RalphLoopsAdapter(ralph_loops),
    }
