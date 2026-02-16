"""OpenCode transport orchestration.

Owns the HTTP client session wiring and background tasks.
Parsing/state updates live in OpenCodeEventProcessor.
"""

from __future__ import annotations

import asyncio
import logging
import os

import aiohttp
from src.runners.base import RunState
from src.runners.opencode.client import OpenCodeClient
from src.runners.opencode.processor import OpenCodeEventProcessor

log = logging.getLogger("opencode")


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
                self._client.abort_session(
                    self._client_session, self._active_session_id
                )
            )

    async def wait_cancelled(self) -> None:
        if self._abort_task:
            try:
                await self._abort_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                msg = str(e).lower()
                if "connector is closed" in msg or "server disconnected" in msg:
                    log.debug(f"OpenCode abort task ended during shutdown: {e}")
                else:
                    log.warning(f"OpenCode abort task failed during wait_cancelled: {e}")

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
            try:
                await sse_task
            except asyncio.CancelledError:
                pass

        if message_task:
            if not message_task.done():
                message_task.cancel()
            try:
                await message_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                # Cancellation commonly races with server-side disconnects or
                # aborted HTTP responses. Consume these exceptions here so they
                # don't surface later as "Task exception was never retrieved".
                msg = str(e).lower()
                if "server disconnected" in msg or "connector is closed" in msg:
                    log.debug(f"OpenCode message task ended during cleanup: {e}")
                elif self._cancelled:
                    log.debug(f"OpenCode message task failed after cancel: {e}")
                else:
                    raise

        if (
            self._client_session
            and self._active_session_id
            and not self._client_session.closed
        ):
            try:
                await self._client.abort_session(
                    self._client_session, self._active_session_id
                )
            except Exception as e:
                msg = str(e).lower()
                if "connector is closed" in msg or "server disconnected" in msg:
                    log.debug(f"OpenCode abort during cleanup ended after disconnect: {e}")
                else:
                    log.warning(f"OpenCode abort failed during cleanup: {e}")

        if self._abort_task:
            try:
                await self._abort_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                # Avoid noisy "Task exception was never retrieved" warnings when
                # cancel races with session/connector shutdown.
                msg = str(e).lower()
                if "connector is closed" in msg or "server disconnected" in msg:
                    log.debug(f"OpenCode abort task ended during cleanup: {e}")
                else:
                    log.warning(f"OpenCode abort task failed during cleanup: {e}")

        self._client_session = None
        self._abort_task = None


def build_http_timeout(*, total_s: float | None = None) -> aiohttp.ClientTimeout:
    http_timeout_s = (
        float(total_s)
        if total_s is not None
        else float(os.getenv("OPENCODE_HTTP_TIMEOUT_S", "600"))
    )
    return aiohttp.ClientTimeout(total=http_timeout_s)
