"""OpenCode run orchestration helpers.

This module owns the event-loop glue between:
- SSE streaming task feeding an event queue
- message POST task
- event parsing + state updates
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import AsyncIterator, Awaitable, Callable

import aiohttp

from src.runners.base import RunState
from src.runners.opencode.events import extract_session_id
from src.runners.opencode.models import Event, Question
from src.runners.opencode.processor import OpenCodeEventProcessor


async def iter_opencode_events(
    *,
    session: aiohttp.ClientSession,
    session_id: str,
    state: RunState,
    event_queue: asyncio.Queue[dict],
    sse_task: asyncio.Task,
    message_task: asyncio.Task,
    processor: OpenCodeEventProcessor,
    should_cancel: Callable[[], bool],
    handle_question: Callable[[aiohttp.ClientSession, Question], Awaitable[None]],
) -> AsyncIterator[Event]:
    """Yield parsed events until completion or cancellation."""

    idle_timeout_s = float(os.getenv("OPENCODE_POST_MESSAGE_IDLE_TIMEOUT_S", "30"))
    message_done_at: float | None = None
    last_event_at = time.monotonic()

    while True:
        if should_cancel():
            break

        if sse_task.done() and not sse_task.cancelled():
            exc = sse_task.exception()
            if exc:
                raise exc

        if message_task.done() and (state.saw_result or state.saw_error):
            break

        if message_task.done() and message_done_at is None:
            message_done_at = time.monotonic()

        if message_done_at is not None and not state.saw_result and not state.saw_error:
            if (time.monotonic() - last_event_at) >= idle_timeout_s:
                break

        try:
            payload = await asyncio.wait_for(event_queue.get(), timeout=0.25)
        except asyncio.TimeoutError:
            continue

        if not isinstance(payload, dict):
            continue

        payload_session = extract_session_id(payload)
        if payload_session and payload_session != session_id:
            continue

        result = processor.parse_event(payload, state)
        if not result:
            continue

        last_event_at = time.monotonic()

        event_type, data = result
        if event_type == "question" and isinstance(data, Question):
            yield result
            await handle_question(session, data)
        else:
            yield result
