"""OpenCode server runner using HTTP + SSE."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import AsyncIterator

import aiohttp

from src.runners.base import BaseRunner, RunState
from src.runners.opencode.client import OpenCodeClient
from src.runners.opencode.config import OpenCodeConfig
from src.runners.opencode.events import extract_session_id
from src.runners.opencode.models import Event, Question, QuestionCallback
from src.runners.opencode.processor import OpenCodeEventProcessor
from src.runners.opencode.transport import OpenCodeTransport, build_http_timeout
from src.runners.pipeline import iter_queue_pipeline

log = logging.getLogger("opencode")


class _QuestionHandler:
    async def handle(
        self,
        session: aiohttp.ClientSession,
        client: OpenCodeClient,
        question: Question,
        *,
        cancelled,
    ) -> None:
        raise NotImplementedError


class _RejectQuestionHandler(_QuestionHandler):
    async def handle(
        self,
        session: aiohttp.ClientSession,
        client: OpenCodeClient,
        question: Question,
        *,
        cancelled,
    ) -> None:
        # Always make forward progress: if the higher-level app isn't wired to
        # answer questions, reject them immediately.
        await client.reject_question(session, question)


class _CallbackQuestionHandler(_QuestionHandler):
    def __init__(self, callback: QuestionCallback):
        self._callback = callback

    async def handle(
        self,
        session: aiohttp.ClientSession,
        client: OpenCodeClient,
        question: Question,
        *,
        cancelled,
    ) -> None:
        callback_task = asyncio.create_task(self._callback(question))
        answered = False
        try:
            while not callback_task.done():
                if cancelled():
                    callback_task.cancel()
                    raise asyncio.CancelledError()
                await asyncio.sleep(0.1)
            answers = callback_task.result()
            await client.answer_question(session, question, answers)
            answered = True
        finally:
            if not answered:
                await client.reject_question(session, question)


class OpenCodeRunner(BaseRunner):
    """Runs OpenCode via the server API with SSE streaming.

    Microdirective: set OPENCODE_PERMISSION='{"*":"allow"}' on the server to
    auto-approve permissions and avoid permission prompts in server mode.
    """

    def __init__(
        self,
        working_dir: str,
        output_dir: Path,
        session_name: str | None = None,
        config: OpenCodeConfig | None = None,
    ):
        super().__init__(working_dir, output_dir, session_name)

        self._config = config or OpenCodeConfig()

        self._client = OpenCodeClient(server_url=self._config.server_url)
        self._processor = OpenCodeEventProcessor(
            log_to_file=self._log_to_file,
            log_response=self._log_response,
            model=self._config.model,
        )
        self._transport = OpenCodeTransport(self._client)

        self._question_handler: _QuestionHandler
        if self._config.question_callback:
            self._question_handler = _CallbackQuestionHandler(
                self._config.question_callback
            )
        else:
            self._question_handler = _RejectQuestionHandler()

    def _build_model_payload(self) -> dict | None:
        if not self._config.model:
            return None
        if "/" not in self._config.model:
            return None
        provider_id, model_id = self._config.model.split("/", 1)
        if not provider_id or not model_id:
            return None
        return {"providerID": provider_id, "modelID": model_id}

    async def _handle_question_event(
        self,
        session: aiohttp.ClientSession,
        question: Question,
    ) -> None:
        await self._question_handler.handle(
            session,
            self._client,
            question,
            cancelled=lambda: self._transport.cancelled,
        )

    async def run(
        self,
        prompt: str,
        session_id: str | None = None,
    ) -> AsyncIterator[Event]:
        """Run OpenCode, yielding (event_type, content) tuples.

        Events:
            ("session_id", str) - Session ID for continuity
            ("text", str) - Incremental response text
            ("tool", str) - Tool invocation description
            ("question", Question) - Question from AI needing answer
            ("result", dict) - Final result stats payload
            ("error", str) - Error message
        """
        state = RunState()
        log.info(f"OpenCode: {prompt[:50]}...")
        self._log_prompt(prompt)

        sse_task: asyncio.Task | None = None
        message_task: asyncio.Task | None = None
        event_queue: asyncio.Queue[dict] = asyncio.Queue()

        try:
            async with aiohttp.ClientSession(
                auth=self._client.auth,
                timeout=build_http_timeout(total_s=self._config.http_timeout_s),
            ) as session:
                session_id = await self._transport.start_session(
                    session,
                    session_name=self.session_name,
                    session_id=session_id,
                )
                state.session_id = session_id
                yield ("session_id", session_id)

                sse_task, message_task = self._transport.start_tasks(
                    session,
                    session_id=session_id,
                    prompt=prompt,
                    model_payload=self._build_model_payload(),
                    agent=self._config.agent,
                    reasoning_mode=self._config.reasoning_mode,
                    event_queue=event_queue,
                )

                async def _handle_question(e: Event) -> None:
                    _, data = e
                    if isinstance(data, Question):
                        await self._handle_question_event(session, data)

                def _is_question(e: Event) -> bool:
                    event_type, data = e
                    return event_type == "question" and isinstance(data, Question)

                if self._config.post_message_idle_timeout_s is not None:
                    idle_timeout_s = float(self._config.post_message_idle_timeout_s)
                else:
                    idle_timeout_s = float(
                        os.getenv("OPENCODE_POST_MESSAGE_IDLE_TIMEOUT_S", "30")
                    )

                async for event in iter_queue_pipeline(
                    event_queue=event_queue,
                    session_id=session_id,
                    state=state,
                    parse_event=self._processor.parse_event,
                    extract_session_id=extract_session_id,
                    sse_task=sse_task,
                    message_task=message_task,
                    should_cancel=lambda: self._transport.cancelled,
                    idle_timeout_s=idle_timeout_s,
                    is_done=lambda s: s.saw_result or s.saw_error,
                    is_question=_is_question,
                    handle_question=_handle_question,
                ):
                    yield event

                # If cancellation was requested, don't wait on the message POST or poll.
                # Cleanup will cancel in-flight tasks and send an abort to the server.
                if self._transport.cancelled:
                    yield ("cancelled", None)
                    return

                if not state.saw_result and not state.saw_error:
                    result = await self._transport.finalize(
                        session=session,
                        session_id=session_id,
                        state=state,
                        message_task=message_task,
                        processor=self._processor,
                    )
                    yield ("result", result)
        finally:
            await self._transport.cleanup(sse_task=sse_task, message_task=message_task)

    def cancel(self) -> None:
        """Request cancellation of the running session."""
        self._transport.cancel()

    async def wait_cancelled(self) -> None:
        """Wait for cancellation cleanup to complete."""
        await self._transport.wait_cancelled()
