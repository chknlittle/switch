"""OpenCode event processing.

The runner orchestrates HTTP/SSE tasks and cancellation.
This module focuses on:
- normalizing/coercing event payloads into yielded Events
- updating RunState
- formatting tool log output consistently
"""

from __future__ import annotations

from typing import Callable

from src.runners.base import RunState
from src.runners.opencode import event_handlers
from src.runners.opencode.events import coerce_event
from src.runners.opencode.models import Event
from src.runners.opencode.text_sanitize import sanitize_assistant_text


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

    @staticmethod
    def _sanitize_assistant_text(text: str) -> str:
        return sanitize_assistant_text(text)

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

        log = self._log_to_file

        if event_type == "step_start":
            return event_handlers.handle_step_start(event, state)
        if event_type == "text":
            return event_handlers.handle_text(event, state)
        if event_type == "tool_use":
            return event_handlers.handle_tool_use(event, state, log_to_file=log)
        if event_type == "tool_result":
            return event_handlers.handle_tool_result(event, state, log_to_file=log)
        if event_type == "step_finish":
            return event_handlers.handle_step_finish(
                event, state, make_result=self.make_result
            )
        if event_type == "error":
            return event_handlers.handle_error(event, state)
        if event_type in {"question.asked", "question"}:
            return event_handlers.handle_question(event, state, log_to_file=log)
        if event_type == "permission.requested":
            return event_handlers.handle_permission_request(
                event, state, log_to_file=log
            )
        if event_type == "message_meta":
            return event_handlers.handle_message_meta(event, state)
        if event_type == "message_part_meta":
            return event_handlers.handle_message_part_meta(event, state)
        if event_type == "message_part_delta":
            return event_handlers.handle_message_part_delta(event, state)
        if event_type == "message_part":
            return event_handlers.handle_message_part(event, state)

        return None
