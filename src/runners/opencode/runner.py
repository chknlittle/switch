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
from src.runners.opencode.models import Event, Question, QuestionCallback
from src.runners.opencode.orchestrator import iter_opencode_events
from src.runners.opencode.processor import OpenCodeEventProcessor

from src.attachments import Attachment

log = logging.getLogger("opencode")


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
        model: str | None = None,
        reasoning_mode: str = "normal",
        agent: str = "bridge",
        question_callback: QuestionCallback | None = None,
        server_url: str | None = None,
    ):
        super().__init__(working_dir, output_dir, session_name)
        self.model = model
        self.reasoning_mode = reasoning_mode
        self.agent = agent
        self.question_callback = question_callback
        self._client = OpenCodeClient(server_url=server_url)
        self._processor = OpenCodeEventProcessor(
            log_to_file=self._log_to_file,
            log_response=self._log_response,
        )
        self._client_session: aiohttp.ClientSession | None = None
        self._active_session_id: str | None = None
        self._cancelled = False
        self._abort_task: asyncio.Task | None = None

    def _build_model_payload(self) -> dict | None:
        if not self.model:
            return None
        if "/" not in self.model:
            return None
        provider_id, model_id = self.model.split("/", 1)
        if not provider_id or not model_id:
            return None
        return {"providerID": provider_id, "modelID": model_id}

    async def _run_question_callback(self, question: Question) -> list[list[str]]:
        if not self.question_callback:
            return []
        return await self.question_callback(question)

    async def _handle_question_event(
        self,
        session: aiohttp.ClientSession,
        question: Question,
    ) -> None:
        """Handle question event - run callback and answer/reject."""
        if not self.question_callback:
            return

        callback_task = asyncio.create_task(self._run_question_callback(question))
        answered = False
        try:
            while not callback_task.done():
                if self._cancelled:
                    callback_task.cancel()
                    raise asyncio.CancelledError()
                await asyncio.sleep(0.1)
            answers = callback_task.result()
            await self._client.answer_question(session, question, answers)
            answered = True
        finally:
            if not answered:
                await self._client.reject_question(session, question)

    async def _cleanup_tasks(
        self, sse_task: asyncio.Task | None, message_task: asyncio.Task | None
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
            await self._client.abort_session(
                self._client_session, self._active_session_id
            )

        if self._abort_task and not self._abort_task.done():
            await self._abort_task
        self._client_session = None
        self._abort_task = None

    async def run(
        self,
        prompt: str,
        session_id: str | None = None,
        *,
        attachments: list[Attachment] | None = None,
    ) -> AsyncIterator[Event]:
        """Run OpenCode, yielding (event_type, content) tuples.

        Events:
            ("session_id", str) - Session ID for continuity
            ("text", str) - Incremental response text
            ("tool", str) - Tool invocation description
            ("question", Question) - Question from AI needing answer
            ("result", OpenCodeResult) - Final result with stats
            ("error", str) - Error message
        """
        state = RunState()
        log.info(f"OpenCode: {prompt[:50]}...")
        self._log_prompt(prompt)

        sse_task: asyncio.Task | None = None
        message_task: asyncio.Task | None = None
        event_queue: asyncio.Queue[dict] = asyncio.Queue()

        try:
            # Some OpenCode server modes keep /message open until completion.
            # Make the HTTP client timeout configurable so long/slow sessions
            # don't fail at a hard-coded limit.
            http_timeout_s = float(os.getenv("OPENCODE_HTTP_TIMEOUT_S", "600"))
            timeout = aiohttp.ClientTimeout(total=http_timeout_s)
            async with aiohttp.ClientSession(auth=self._client.auth, timeout=timeout) as session:
                self._client_session = session
                await self._client.check_health(session)

                if not session_id:
                    session_id = await self._client.create_session(session, self.session_name)

                state.session_id = session_id
                self._active_session_id = session_id
                yield ("session_id", session_id)

                sse_task = asyncio.create_task(
                    self._client.stream_events(session, event_queue, should_stop=lambda: self._cancelled)
                )
                message_task = asyncio.create_task(
                    self._client.send_message(
                        session,
                        session_id,
                        prompt,
                        attachments,
                        self._build_model_payload(),
                        self.agent,
                        self.reasoning_mode,
                    )
                )

                async for event in iter_opencode_events(
                    session=session,
                    session_id=session_id,
                    state=state,
                    event_queue=event_queue,
                    sse_task=sse_task,
                    message_task=message_task,
                    processor=self._processor,
                    should_cancel=lambda: self._cancelled,
                    handle_question=self._handle_question_event,
                ):
                    yield event

                response = await message_task
                if isinstance(response, dict):
                    self._processor.process_message_response(response, state)
                    if not state.saw_result:
                        state.saw_result = True
                        yield ("result", self._processor.make_result(state))
                elif not state.saw_result and not state.saw_error:
                    # Some server modes return an empty body for /message and rely on
                    # storing output in the session message list. Fall back to polling.
                    polled = await self._client.poll_assistant_text(session, session_id)
                    if polled and isinstance(polled, str):
                        state.text = polled
                        state.saw_result = True
                        yield ("result", self._processor.make_result(state))
                    else:
                        event_type, message = self._processor.make_fallback_error(state)
                        raise RuntimeError(message)
        finally:
            await self._cleanup_tasks(sse_task, message_task)

    def cancel(self) -> None:
        """Request cancellation of the running session."""
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
        """Wait for cancellation cleanup to complete."""
        if self._abort_task:
            await self._abort_task
