"""OpenCode event processing.

The runner orchestrates HTTP/SSE tasks and cancellation.
This module focuses on:
- normalizing/coercing event payloads into yielded Events
- updating RunState
- formatting tool log output consistently
"""

from __future__ import annotations

import os
from typing import Callable

from src.runners.base import RunState
from src.runners.opencode.events import coerce_event
from src.runners.opencode.models import Event, Question
from src.runners.tool_logging import (
    format_tool_input_preview,
    should_log_tool_input,
    tool_input_max_len,
)


class OpenCodeEventProcessor:
    def __init__(
        self,
        *,
        log_to_file: Callable[[str], None],
        log_response: Callable[[str], None] | None = None,
        model: str | None = None,
    ):
        self._log_to_file = log_to_file
        self._log_response = log_response
        self._model = model

    def _handle_step_start(self, event: dict, state: RunState) -> Event | None:
        session_id = event.get("sessionID")
        if isinstance(session_id, str) and session_id:
            state.session_id = session_id
            return ("session_id", session_id)
        return None

    def _apply_text_update(self, text: str, state: RunState) -> Event | None:
        if not text:
            return None

        # SSE sends full accumulated text, not deltas - extract only the new part.
        if text.startswith(state.text):
            delta = text[len(state.text) :]
            state.text = text
            if delta:
                return ("text", delta)
            return None

        state.text = text
        return ("text", text)

    def _handle_text(self, event: dict, state: RunState) -> Event | None:
        part = event.get("part", {})
        text = part.get("text", "") if isinstance(part, dict) else ""
        if isinstance(text, str):
            return self._apply_text_update(text, state)
        return None

    def _handle_message_meta(self, event: dict, state: RunState) -> Event | None:
        message_id = event.get("messageID")
        role = event.get("role")
        if isinstance(message_id, str) and message_id and isinstance(role, str) and role:
            state.message_roles[message_id] = role
        return None

    def _handle_tool_use(self, event: dict, state: RunState) -> Event | None:
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

        if should_log_tool_input():
            raw_input: object | None = None
            if isinstance(tool_state, dict):
                raw_input = tool_state.get("input") or tool_state.get("args")
            if raw_input is None:
                raw_input = part.get("input") or part.get("args")

            formatted = format_tool_input_preview(str(tool), raw_input)
            if formatted:
                max_len = tool_input_max_len()
                formatted = formatted[:max_len]
                self._log_to_file(f"{desc}\n  input: {formatted}\n")
                return ("tool", f"{desc} input: {formatted}")

        self._log_to_file(f"{desc}\n")
        return ("tool", desc)

    def _handle_step_finish(self, event: dict, state: RunState) -> Event | None:
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
            return ("result", self.make_result(state))
        return None

    def _handle_error(self, event: dict, state: RunState) -> Event:
        state.saw_error = True
        message = event.get("message")
        error = event.get("error")

        if isinstance(message, dict):
            message = message.get("data", {}).get("message") or message.get("message")

        return ("error", str(message or error or "OpenCode error"))

    def _handle_question(self, event: dict, state: RunState) -> Event | None:
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
            self._log_to_file(f"Question event missing request ID: {event}\n")
            return None

        question = Question(request_id=request_id, questions=questions)
        self._log_to_file(f"\n[QUESTION] {request_id}: {questions}\n")
        return ("question", question)

    def make_result(self, state: RunState) -> dict:
        if self._log_response and state.text:
            self._log_response(state.text)

        model_short = "?"
        if self._model:
            model_short = self._model.split("/", 1)[-1] or "?"

        return {
            "engine": "opencode",
            "model": model_short,
            "session_id": state.session_id,
            "tool_count": state.tool_count,
            "tokens_in": state.tokens_in,
            "tokens_out": state.tokens_out,
            "tokens_reasoning": state.tokens_reasoning,
            "tokens_cache_read": state.tokens_cache_read,
            "tokens_cache_write": state.tokens_cache_write,
            "cost_usd": float(state.cost),
            "duration_s": float(state.duration_s),
            "text": state.text,
            "summary": (
                f"[{model_short} {state.tokens_in}/{state.tokens_out} tok"
                f" r{state.tokens_reasoning} c{state.tokens_cache_read}/{state.tokens_cache_write}"
                f" ${state.cost:.3f} {state.duration_s:.1f}s]"
            ),
        }

    def process_message_response(self, response: dict, state: RunState) -> None:
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

    def make_fallback_error(self, state: RunState) -> Event:
        if state.raw_output:
            preview = " | ".join(state.raw_output)
            return ("error", f"OpenCode output (non-JSON): {preview}")
        return ("error", "OpenCode exited without output")

    def parse_event(self, raw_event: dict, state: RunState) -> Event | None:
        event = coerce_event(raw_event)
        if not event:
            return None

        event_type = event.get("type")
        if not isinstance(event_type, str):
            return None

        if event_type == "step_start":
            return self._handle_step_start(event, state)
        if event_type == "text":
            return self._handle_text(event, state)
        if event_type == "tool_use":
            return self._handle_tool_use(event, state)
        if event_type == "step_finish":
            return self._handle_step_finish(event, state)
        if event_type == "error":
            return self._handle_error(event, state)
        if event_type in {"question.asked", "question"}:
            return self._handle_question(event, state)
        if event_type == "message_meta":
            return self._handle_message_meta(event, state)

        # Server-mode streams often send message events rather than "text".
        if event_type == "message_part":
            message_id = event.get("messageID")
            role = (
                state.message_roles.get(message_id)
                if isinstance(message_id, str) and message_id
                else None
            )
            if role != "assistant":
                return None

            text = event.get("text", "")
            if isinstance(text, str):
                return self._apply_text_update(text, state)

        return None
