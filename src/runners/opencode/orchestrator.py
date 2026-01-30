"""OpenCode run orchestration helpers.

This module owns the event-loop glue between:
- SSE streaming task feeding an event queue
- message POST task
- event parsing + state updates
"""

from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator, Awaitable, Callable

import aiohttp

from src.runners.base import RunState
from src.runners.opencode.models import Event, Question
from src.runners.opencode.processor import OpenCodeEventProcessor
from src.runners.opencode.events import extract_session_id
from src.runners.pipeline import iter_queue_pipeline


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
    idle_timeout_s = float(os.getenv("OPENCODE_POST_MESSAGE_IDLE_TIMEOUT_S", "30"))

    async def _handle_question_event(e: Event) -> None:
        _, data = e
        if isinstance(data, Question):
            await handle_question(session, data)

    def _is_question(e: Event) -> bool:
        event_type, data = e
        return event_type == "question" and isinstance(data, Question)

    async for e in iter_queue_pipeline(
        event_queue=event_queue,
        session_id=session_id,
        state=state,
        parse_event=processor.parse_event,
        extract_session_id=extract_session_id,
        sse_task=sse_task,
        message_task=message_task,
        should_cancel=should_cancel,
        idle_timeout_s=idle_timeout_s,
        is_done=lambda s: s.saw_result or s.saw_error,
        is_question=_is_question,
        handle_question=_handle_question_event,
    ):
        yield e
