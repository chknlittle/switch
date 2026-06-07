"""Cursor ACP event processing."""

from __future__ import annotations

from typing import Any, Callable

from src.runners.base import RunState
from src.runners.ports import RunnerEvent


Event = RunnerEvent


class CursorEventProcessor:
    def __init__(
        self,
        *,
        log_response: Callable[[str], None] | None = None,
    ):
        self._log_response = log_response

    @staticmethod
    def _tool_summary(update: dict[str, Any]) -> str | None:
        kind = update.get("sessionUpdate")
        if kind in {"tool_call", "tool_call_update", "tool_call_started"}:
            name = update.get("name") or update.get("toolName") or update.get("title") or "tool"
            return f"Cursor: {name}"
        if kind in {"command_started", "command_update"}:
            cmd = update.get("command") or update.get("text") or "command"
            return f"Cursor shell: {cmd}"
        return None

    def parse_event(
        self,
        msg: dict[str, Any],
        state: RunState,
        *,
        cursor_session_id: str,
    ) -> Event | list[Event] | None:
        method = msg.get("method")
        if method == "session/request_permission":
            return ("permission", msg)

        if method != "session/update":
            return None

        params = msg.get("params") or {}
        if not isinstance(params, dict):
            return None

        payload_session = params.get("sessionId")
        if payload_session not in {None, cursor_session_id}:
            return None

        update = params.get("update") or {}
        if not isinstance(update, dict):
            return None

        kind = update.get("sessionUpdate")
        if kind == "agent_message_chunk":
            content = update.get("content") or {}
            text = content.get("text") if isinstance(content, dict) else ""
            if isinstance(text, str) and text:
                state.text += text
                return ("text", text)
            return None

        if kind == "agent_thought_chunk":
            return None

        summary = self._tool_summary(update)
        if summary:
            return ("tool", summary)
        return None

    def make_result(self, state: RunState, prompt_result: object) -> dict[str, Any]:
        if self._log_response and state.text:
            self._log_response(state.text)

        stop_reason = None
        if isinstance(prompt_result, dict):
            stop_reason = prompt_result.get("stopReason")

        return {
            "duration_s": state.duration_s,
            "session_id": state.session_id,
            "stop_reason": stop_reason,
        }
