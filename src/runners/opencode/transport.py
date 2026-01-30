"""OpenCode transport orchestration.

Owns the HTTP client session wiring and background tasks.
Parsing/state updates live in OpenCodeEventProcessor.
"""

from __future__ import annotations

import asyncio
import os

import aiohttp

from src.attachments import Attachment
from src.runners.base import RunState
from src.runners.opencode.client import OpenCodeClient
from src.runners.opencode.processor import OpenCodeEventProcessor


class OpenCodeTransport:
    def __init__(self, client: OpenCodeClient):
        self._client = client
        self._client_session: aiohttp.ClientSession | None = None
        self._active_session_id: str | None = None
        self._cancelled = False
        self._abort_task: asyncio.Task | None = None

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True
        if (
            self._client_session
            and self._active_session_id
            and not self._client_session.closed
        ):
            self._abort_task = asyncio.create_task(
                self._client.abort_session(self._client_session, self._active_session_id)
            )

    async def wait_cancelled(self) -> None:
        if self._abort_task:
            await self._abort_task

    async def start_session(
        self,
        session: aiohttp.ClientSession,
        *,
        session_name: str | None,
        session_id: str | None,
    ) -> str:
        self._client_session = session
        await self._client.check_health(session)
        if not session_id:
            session_id = await self._client.create_session(session, session_name)
        self._active_session_id = session_id
        return session_id

    def start_tasks(
        self,
        session: aiohttp.ClientSession,
        *,
        session_id: str,
        prompt: str,
        attachments: list[Attachment] | None,
        model_payload: dict | None,
        agent: str,
        reasoning_mode: str,
        event_queue: asyncio.Queue[dict],
    ) -> tuple[asyncio.Task, asyncio.Task]:
        sse_task = asyncio.create_task(
            self._client.stream_events(
                session, event_queue, should_stop=lambda: self._cancelled
            )
        )
        message_task = asyncio.create_task(
            self._client.send_message(
                session,
                session_id,
                prompt,
                attachments,
                model_payload,
                agent,
                reasoning_mode,
            )
        )
        return sse_task, message_task

    async def finalize(
        self,
        *,
        session: aiohttp.ClientSession,
        session_id: str,
        state: RunState,
        message_task: asyncio.Task,
        processor: OpenCodeEventProcessor,
    ) -> dict:
        response = await message_task
        if isinstance(response, dict):
            processor.process_message_response(response, state)
            state.saw_result = True
            return processor.make_result(state)

        if not state.saw_result and not state.saw_error:
            polled = await self._client.poll_assistant_text(session, session_id)
            if polled and isinstance(polled, str):
                state.text = polled
                state.saw_result = True
                return processor.make_result(state)

            _, message = processor.make_fallback_error(state)
            raise RuntimeError(message)

        return processor.make_result(state)

    async def cleanup(
        self, *, sse_task: asyncio.Task | None, message_task: asyncio.Task | None
    ) -> None:
        """Cleanup.

        This project prefers bubbling errors for simplicity; cleanup is not
        guaranteed to be best-effort.
        """

        self._cancelled = True

        if sse_task:
            sse_task.cancel()
            await sse_task

        if message_task and not message_task.done():
            message_task.cancel()
            await message_task

        if (
            self._client_session
            and self._active_session_id
            and not self._client_session.closed
        ):
            await self._client.abort_session(self._client_session, self._active_session_id)

        if self._abort_task and not self._abort_task.done():
            await self._abort_task

        self._client_session = None
        self._abort_task = None


def build_http_timeout() -> aiohttp.ClientTimeout:
    http_timeout_s = float(os.getenv("OPENCODE_HTTP_TIMEOUT_S", "600"))
    return aiohttp.ClientTimeout(total=http_timeout_s)
