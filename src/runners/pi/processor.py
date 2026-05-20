"""Pi RPC event processing.

Maps pi's RPC event stream to the standard RunnerEvent tuples that switch
expects: ("session_id"|"text"|"tool"|"tool_result"|"result"|"error", data).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.runners.base import RunState
from src.runners.ports import RunnerEvent

Event = RunnerEvent


def _assistant_terminal_error(message: object) -> str | None:
    """Extract a user-visible error from a Pi assistant message."""
    if not isinstance(message, dict):
        return None

    inner = message.get("message")
    msg = inner if isinstance(inner, dict) else message
    if msg.get("role") != "assistant":
        return None

    stop = msg.get("stopReason")
    if stop not in ("error", "aborted"):
        return None

    err = msg.get("errorMessage")
    if isinstance(err, str) and err.strip():
        return err.strip()
    if err is not None:
        return str(err)
    if stop == "aborted":
        return "Request was aborted"
    return "Assistant request failed"


def _terminal_error_from_messages(messages: object) -> str | None:
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        err = _assistant_terminal_error(message)
        if err:
            return err
    return None


class PiEventProcessor:
    def __init__(
        self,
        *,
        log_to_file: Callable[[str], None],
        log_response: Callable[[str], None],
    ):
        self._log_to_file = log_to_file
        self._log_response = log_response

    def _handle_message_update(self, event: dict, state: RunState) -> Event | None:
        ame = event.get("assistantMessageEvent")
        if not isinstance(ame, dict):
            return None

        ame_type = ame.get("type")

        if ame_type == "text_delta":
            delta = ame.get("delta", "") or ame.get("text", "")
            if delta:
                state.text += delta
                return ("text", delta)

        if ame_type == "toolcall_start":
            state.tool_count += 1
            # Tool name is nested in partial.content[].name
            name = "?"
            partial = ame.get("partial", {})
            for block in (partial.get("content") or []):
                if isinstance(block, dict) and block.get("type") == "toolCall":
                    name = block.get("name", "?")
                    break
            desc = f"[tool:{name}]"
            self._log_to_file(f"{desc}\n")
            return ("tool", desc)

        if ame_type == "toolcall_delta":
            # Streaming tool arguments — skip for now, we get full info at execution.
            pass

        if ame_type == "error":
            err_obj = ame.get("error")
            error = ame.get("errorMessage")
            if isinstance(err_obj, dict):
                error = error or err_obj.get("errorMessage")
            if not error:
                error = _assistant_terminal_error(
                    event.get("message") if isinstance(event.get("message"), dict) else {}
                )
            state.saw_error = True
            state.terminal_error = str(error or "Pi error")
            return ("error", state.terminal_error)

        return None

    def _handle_tool_execution(self, event: dict, state: RunState) -> Event | None:
        event_type = event.get("type")

        if event_type == "tool_execution_start":
            name = event.get("toolName", "?")
            args = event.get("args") or event.get("arguments")

            # Build a useful description.
            extra = ""
            if isinstance(args, dict):
                if name == "bash" and "command" in args:
                    cmd = str(args["command"]).strip()
                    if len(cmd) > 80:
                        cmd = cmd[:77] + "..."
                    extra = f" {cmd}"
                elif name in ("read", "write", "edit") and "file_path" in args:
                    leaf = Path(str(args["file_path"])).name
                    extra = f" {leaf}"

            desc = f"[tool:{name}{extra}]"
            self._log_to_file(f"{desc}\n")
            return ("tool", desc)

        if event_type == "tool_execution_end":
            name = event.get("toolName", "?")
            # Compact result summary.
            result_obj = event.get("result", {})
            result_content = result_obj.get("content") if isinstance(result_obj, dict) else event.get("content")
            pieces: list[str] = []

            exit_code = event.get("exitCode")
            if exit_code is not None:
                pieces.append(f"exit={exit_code}")

            if isinstance(result_content, list):
                for part in result_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = str(part.get("text", "")).strip()
                        if text:
                            if len(text) > 180:
                                text = text[:177] + "..."
                            pieces.append(text)
                            break

            is_error = event.get("isError", False)
            if is_error:
                pieces.append("ERROR")

            suffix = f" {' | '.join(pieces)}" if pieces else ""
            desc = f"[tool-result:{name}{suffix}]"
            self._log_to_file(f"{desc}\n")
            return ("tool_result", desc)

        return None

    def _record_terminal_error(self, error: str, state: RunState) -> Event:
        state.saw_error = True
        state.terminal_error = error
        self._log_to_file(f"[error] {error}\n")
        return ("error", error)

    def _handle_agent_end(self, event: dict, state: RunState) -> list[Event]:
        state.saw_result = True

        if self._log_response and state.text:
            self._log_response(state.text)

        events: list[Event] = []
        if not state.text:
            error = _terminal_error_from_messages(event.get("messages"))
            if error:
                events.append(self._record_terminal_error(error, state))
        return events

    def _handle_message_end(self, event: dict, state: RunState) -> list[Event]:
        events: list[Event] = []
        if state.text:
            return events
        error = _assistant_terminal_error(event.get("message"))
        if error:
            events.append(self._record_terminal_error(error, state))
        return events

    def make_result(self, state: RunState, stats: dict | None = None) -> dict:
        if self._log_response and state.text:
            self._log_response(state.text)

        usage = stats or {}
        # Pi nests token counts under a "tokens" sub-object:
        # {"tokens": {"input": N, "output": N, ...}, "cost": ..., "model": ...}
        tokens = usage.get("tokens", {}) if isinstance(usage, dict) else {}
        if not isinstance(tokens, dict):
            tokens = {}
        # Fall back to top-level keys for backwards compat.
        tokens_in = int(tokens.get("input", 0) or usage.get("input", 0) or 0)
        tokens_out = int(tokens.get("output", 0) or usage.get("output", 0) or 0)
        tokens_cache_read = int(tokens.get("cacheRead", 0) or usage.get("cacheRead", 0) or 0)
        tokens_cache_write = int(tokens.get("cacheWrite", 0) or usage.get("cacheWrite", 0) or 0)
        tokens_total = int(tokens.get("total", 0) or usage.get("total", 0) or 0)

        cost_info = usage.get("cost", {})
        cost_usd = 0.0
        if isinstance(cost_info, dict):
            for v in cost_info.values():
                if isinstance(v, (int, float)):
                    cost_usd += float(v)
        elif isinstance(cost_info, (int, float)):
            cost_usd = float(cost_info)

        model = str(usage.get("model", "pi") or "pi")

        return {
            "engine": "pi",
            "model": model,
            "session_id": state.session_id,
            "tool_count": state.tool_count,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_reasoning": 0,
            "tokens_cache_read": tokens_cache_read,
            "tokens_cache_write": tokens_cache_write,
            "tokens_total": tokens_total,
            "cost_usd": cost_usd,
            "duration_s": float(state.duration_s),
            "text": state.text,
            "summary": (
                f"[pi {tokens_in}/{tokens_out} tok"
                f" c{tokens_cache_read}/{tokens_cache_write}"
                f" ${cost_usd:.3f} {state.duration_s:.1f}s]"
            ),
        }

    def parse_event(self, event: dict, state: RunState) -> list[Event]:
        event_type = event.get("type")
        if not isinstance(event_type, str):
            return []

        if event_type == "message_update":
            result = self._handle_message_update(event, state)
            return [result] if result else []

        if event_type in ("tool_execution_start", "tool_execution_end"):
            result = self._handle_tool_execution(event, state)
            return [result] if result else []

        if event_type == "agent_end":
            return self._handle_agent_end(event, state)

        if event_type == "message_end":
            return self._handle_message_end(event, state)

        if event_type == "extension_ui_request":
            # Return a signal so the runner can auto-respond via stdin.
            # Fire-and-forget methods need no response.
            method = event.get("method", "")
            if method in ("notify", "setStatus", "setWidget", "setTitle", "set_editor_text"):
                return []
            return [("_extension_ui_request", event)]

        return []
