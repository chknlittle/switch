"""OpenCode server runner using HTTP + SSE."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import AsyncIterator

import aiohttp

from src.runners.base import BaseRunner, RunState
from src.runners.opencode.client import OpenCodeClient
from src.runners.opencode.events import coerce_event, extract_session_id
from src.runners.opencode.models import (
    Event,
    OpenCodeResult,
    Question,
    QuestionCallback,
)

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

    def _handle_step_start(self, event: dict, state: RunState) -> Event | None:
        """Handle step_start event - extracts session ID."""
        session_id = event.get("sessionID")
        if isinstance(session_id, str) and session_id:
            state.session_id = session_id
            return ("session_id", session_id)
        return None

    def _apply_text_update(self, text: str, state: RunState) -> Event | None:
        """Apply accumulated text update and return delta event if any."""
        if not text:
            return None
        # SSE sends full accumulated text, not deltas - extract only the new part.
        if text.startswith(state.text):
            delta = text[len(state.text) :]
            state.text = text
            if delta:
                return ("text", delta)
            return None
        # Text doesn't match previous - might be a new message segment.
        state.text = text
        return ("text", text)

    def _handle_text(self, event: dict, state: RunState) -> Event | None:
        """Handle text event - extracts delta from accumulated text."""
        part = event.get("part", {})
        text = part.get("text", "") if isinstance(part, dict) else ""
        if isinstance(text, str):
            return self._apply_text_update(text, state)
        return None

    def _handle_tool_use(self, event: dict, state: RunState) -> Event | None:
        """Handle tool_use event - tracks tool invocations."""
        part = event.get("part", {})
        if not isinstance(part, dict):
            return None

        tool = part.get("tool")
        if not tool:
            return None

        state.tool_count += 1
        tool_state = part.get("state", {})
        title = tool_state.get("title") if isinstance(tool_state, dict) else None
        desc = f"[tool:{tool} {title}]" if title else f"[tool:{tool}]"

        # Optional: include tool input payload (e.g., bash command) in logs.
        if os.getenv("SWITCH_LOG_TOOL_INPUT", "").lower() in {"1", "true", "yes"}:
            raw_input: object | None = None
            if isinstance(tool_state, dict):
                raw_input = tool_state.get("input") or tool_state.get("args")
            if raw_input is None:
                raw_input = part.get("input") or part.get("args")

            formatted = self._format_tool_input(tool, raw_input)
            if formatted:
                max_len = int(os.getenv("SWITCH_LOG_TOOL_INPUT_MAX", "2000"))
                formatted = formatted[:max_len]
                self._log_to_file(f"{desc}\n  input: {formatted}\n")
                return ("tool", f"{desc} input: {formatted}")

        self._log_to_file(f"{desc}\n")
        return ("tool", desc)

    _REDACT_KEYS = ("key", "token", "secret", "password", "auth", "cookie")

    def _redact_tool_input(self, obj: object) -> object:
        if isinstance(obj, dict):
            out: dict[object, object] = {}
            for k, v in obj.items():
                ks = str(k).lower()
                if any(rk in ks for rk in self._REDACT_KEYS):
                    out[k] = "[REDACTED]"
                else:
                    out[k] = self._redact_tool_input(v)
            return out
        if isinstance(obj, list):
            return [self._redact_tool_input(x) for x in obj]
        return obj

    def _format_tool_input(self, tool: str, raw_input: object) -> str | None:
        if raw_input is None:
            return None

        # Prefer a useful preview for common tools.
        if tool == "bash" and isinstance(raw_input, dict):
            cmd = raw_input.get("command")
            if isinstance(cmd, str) and cmd.strip():
                return cmd.strip()

        if tool in {"read", "write", "edit"} and isinstance(raw_input, dict):
            fp = raw_input.get("filePath") or raw_input.get("file_path")
            if isinstance(fp, str) and fp:
                return fp

        if tool == "grep" and isinstance(raw_input, dict):
            pat = raw_input.get("pattern")
            inc = raw_input.get("include")
            if isinstance(pat, str) and pat:
                suffix = f" include={inc!r}" if isinstance(inc, str) and inc else ""
                return f"pattern={pat!r}" + suffix

        # Fallback: JSON (redacted) for everything else.
        try:
            redacted = self._redact_tool_input(raw_input)
            return json.dumps(redacted, ensure_ascii=True, sort_keys=True)
        except Exception:
            return str(raw_input)

    def _handle_step_finish(self, event: dict, state: RunState) -> Event | None:
        """Handle step_finish event - accumulates tokens/cost, emits result on stop."""
        part = event.get("part", {})
        if not isinstance(part, dict):
            return None

        tokens = part.get("tokens", {})
        if isinstance(tokens, dict):
            cache = tokens.get("cache", {})
            state.tokens_in += int(tokens.get("input", 0) or 0)
            state.tokens_out += int(tokens.get("output", 0) or 0)
            state.tokens_reasoning += int(tokens.get("reasoning", 0) or 0)
            if isinstance(cache, dict):
                state.tokens_cache_read += int(cache.get("read", 0) or 0)
                state.tokens_cache_write += int(cache.get("write", 0) or 0)

        state.cost += float(part.get("cost", 0) or 0)

        if part.get("reason") == "stop":
            state.saw_result = True
            return ("result", self._make_result(state))
        return None

    def _handle_error(self, event: dict, state: RunState) -> Event:
        """Handle error event - extracts error message."""
        state.saw_error = True
        message = event.get("message")
        error = event.get("error")

        if isinstance(message, dict):
            message = message.get("data", {}).get("message") or message.get("message")

        return ("error", str(message or error or "OpenCode error"))

    def _handle_question(self, event: dict, state: RunState) -> Event | None:
        """Handle question.asked event - creates Question object."""
        request_id = (
            event.get("requestID")
            or event.get("id")
            or event.get("properties", {}).get("requestID")
            or event.get("properties", {}).get("id")
        )

        questions = (
            event.get("questions") or event.get("properties", {}).get("questions") or []
        )

        if not request_id:
            log.warning(f"Question event missing request ID: {event}")
            return None

        question = Question(request_id=request_id, questions=questions)
        self._log_to_file(f"\n[QUESTION] {request_id}: {questions}\n")
        return ("question", question)

    def _make_result(self, state: RunState) -> OpenCodeResult:
        """Create result object from current state."""
        if state.text:
            self._log_response(state.text)
        return OpenCodeResult(
            text=state.text,
            session_id=state.session_id,
            cost=state.cost,
            tokens_in=state.tokens_in,
            tokens_out=state.tokens_out,
            tokens_reasoning=state.tokens_reasoning,
            tokens_cache_read=state.tokens_cache_read,
            tokens_cache_write=state.tokens_cache_write,
            duration_s=state.duration_s,
            tool_count=state.tool_count,
        )

    def _parse_event(self, event: dict, state: RunState) -> Event | None:
        """Parse a JSON event and return the appropriate yield value."""
        event_type = event.get("type")
        if not isinstance(event_type, str):
            return None
        handlers = {
            "step_start": self._handle_step_start,
            "text": self._handle_text,
            "tool_use": self._handle_tool_use,
            "step_finish": self._handle_step_finish,
            "error": self._handle_error,
            "question.asked": self._handle_question,
            "question": self._handle_question,
        }

        # Server-mode streams often send message events rather than "text".
        if event_type == "message_part":
            text = event.get("text", "")
            if isinstance(text, str):
                return self._apply_text_update(text, state)

        handler = handlers.get(event_type)
        return handler(event, state) if handler else None

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
        try:
            while not callback_task.done():
                if self._cancelled:
                    callback_task.cancel()
                    raise asyncio.CancelledError()
                await asyncio.sleep(0.1)
            answers = callback_task.result()
            await self._client.answer_question(session, question, answers)
        except asyncio.CancelledError:
            log.info("Question callback cancelled")
            await self._client.reject_question(session, question)
            raise
        except Exception as e:
            log.error(f"Question callback error: {e}")
            await self._client.reject_question(session, question)

    async def _process_event_loop(
        self,
        session: aiohttp.ClientSession,
        session_id: str,
        state: RunState,
        event_queue: asyncio.Queue[dict],
        sse_task: asyncio.Task,
        message_task: asyncio.Task,
    ) -> AsyncIterator[Event]:
        """Process events from SSE stream until completion."""
        # When the message POST returns quickly (often HTTP 204), SSE events may
        # still arrive later. Avoid exiting early just because the POST finished
        # and the queue is momentarily empty.
        idle_timeout_s = float(os.getenv("OPENCODE_POST_MESSAGE_IDLE_TIMEOUT_S", "30"))
        message_done_at: float | None = None
        last_event_at = time.monotonic()

        while True:
            if self._cancelled:
                break
            if sse_task.done() and not sse_task.cancelled():
                exc = sse_task.exception()
                if exc:
                    raise exc
            if message_task.done() and (state.saw_result or state.saw_error):
                break

            if message_task.done() and message_done_at is None:
                message_done_at = time.monotonic()

            if (
                message_done_at is not None
                and not state.saw_result
                and not state.saw_error
            ):
                # If we haven't seen any relevant events for a while after the
                # POST completed, stop waiting and fall back to error handling.
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

            coerced = coerce_event(payload)
            if not coerced:
                continue

            result = self._parse_event(coerced, state)
            if not result:
                continue

            # Only reset idle timer on events that are relevant to this
            # session and produced a meaningful result.
            last_event_at = time.monotonic()

            event_type, data = result
            if event_type == "question" and isinstance(data, Question):
                yield result
                await self._handle_question_event(session, data)
            else:
                yield result

    def _process_message_response(self, response: dict, state: RunState) -> None:
        """Extract text and token info from message response if not seen via SSE."""
        info: dict = {}
        raw_info = response.get("info")
        if isinstance(raw_info, dict):
            info = raw_info

        parts: list = []
        raw_parts = response.get("parts")
        if isinstance(raw_parts, list):
            parts = raw_parts

        if not state.text and parts:
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    if isinstance(text, str):
                        state.text += text

        if state.tokens_in == 0 and state.tokens_out == 0:
            usage = info.get("tokens") or info.get("usage") or {}
            if isinstance(usage, dict):
                cache: dict = {}
                raw_cache = usage.get("cache")
                if isinstance(raw_cache, dict):
                    cache = raw_cache
                state.tokens_in = int(usage.get("input", 0) or 0)
                state.tokens_out = int(usage.get("output", 0) or 0)
                state.tokens_reasoning = int(usage.get("reasoning", 0) or 0)
                state.tokens_cache_read = int(cache.get("read", 0) or 0)
                state.tokens_cache_write = int(cache.get("write", 0) or 0)
            state.cost = float(info.get("cost", 0) or 0)

    async def _cleanup_tasks(
        self, sse_task: asyncio.Task | None, message_task: asyncio.Task | None
    ) -> None:
        """Best-effort cleanup.

        Cleanup must never raise: it's invoked from a `finally:` block and
        any exception here would mask the original failure (e.g. SSE connect
        errors) and/or crash the session loop.
        """
        self._cancelled = True
        if sse_task:
            sse_task.cancel()
            try:
                await sse_task
            except asyncio.CancelledError:
                pass
            except Exception:
                # Never allow cleanup to mask the real error.
                pass
        if (
            self._client_session
            and self._active_session_id
            and not self._client_session.closed
        ):
            try:
                await self._client.abort_session(
                    self._client_session, self._active_session_id
                )
            except Exception:
                # Best-effort.
                pass
        if self._abort_task and not self._abort_task.done():
            try:
                await self._abort_task
            except Exception:
                pass
        self._client_session = None
        self._abort_task = None

    async def run(
        self, prompt: str, session_id: str | None = None
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
            async with aiohttp.ClientSession(
                auth=self._client.auth, timeout=timeout
            ) as session:
                self._client_session = session
                await self._client.check_health(session)

                if not session_id:
                    session_id = await self._client.create_session(
                        session, self.session_name
                    )

                state.session_id = session_id
                self._active_session_id = session_id
                yield ("session_id", session_id)

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
                        self._build_model_payload(),
                        self.agent,
                        self.reasoning_mode,
                    )
                )

                async for event in self._process_event_loop(
                    session, session_id, state, event_queue, sse_task, message_task
                ):
                    yield event

                response = await message_task
                if isinstance(response, dict):
                    self._process_message_response(response, state)
                    if not state.saw_result:
                        state.saw_result = True
                        yield ("result", self._make_result(state))
                elif not state.saw_result and not state.saw_error:
                    # Some server modes return an empty body for /message and rely on
                    # storing output in the session message list. Fall back to polling.
                    polled = await self._client.poll_assistant_text(session, session_id)
                    if polled and isinstance(polled, str):
                        state.text = polled
                        state.saw_result = True
                        yield ("result", self._make_result(state))
                    else:
                        yield self._make_fallback_error(state)

        except asyncio.CancelledError:
            log.info("OpenCode runner was cancelled")
            yield ("cancelled", "OpenCode was cancelled")
        except Exception as e:
            state.saw_error = True
            log.exception(f"OpenCode runner exception: {type(e).__name__}: {e}")
            if isinstance(e, asyncio.TimeoutError):
                # aiohttp can raise a TimeoutError with an empty message.
                # Surface the configured timeout to make this actionable.
                http_timeout_s = float(os.getenv("OPENCODE_HTTP_TIMEOUT_S", "600"))
                yield (
                    "error",
                    f"TimeoutError (HTTP timeout after {http_timeout_s:.0f}s; set OPENCODE_HTTP_TIMEOUT_S to increase)",
                )
            else:
                message = str(e).strip()
                if not message:
                    message = type(e).__name__
                yield ("error", message)
        finally:
            try:
                await self._cleanup_tasks(sse_task, message_task)
            except Exception:
                # Cleanup is best-effort; never raise from finally.
                pass

    def _make_fallback_error(self, state: RunState) -> Event:
        """Create an error event when OpenCode exits without proper result."""
        if state.raw_output:
            preview = " | ".join(state.raw_output)
            return ("error", f"OpenCode output (non-JSON): {preview}")
        return ("error", "OpenCode exited without output")

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
            try:
                await self._abort_task
            except Exception:
                pass
