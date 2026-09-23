"""Pi RPC event processing.

Maps pi's RPC event stream to the standard RunnerEvent tuples that switch
expects: ("session_id"|"text"|"tool"|"tool_result"|"result"|"error", data).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from src.runners.base import RunState
from src.runners.ports import RunnerEvent, ToolResult

Event = RunnerEvent

log = logging.getLogger("pi")

_ABORTED_MSG = "Request was aborted"
_LENGTH_MSG = "Assistant response reached the output token limit"
_NOISE_TERMINAL_ERRORS = frozenset({_ABORTED_MSG, _LENGTH_MSG})


def _assistant_terminal_error(message: object) -> str | None:
    """Extract a user-visible error from a Pi assistant message."""
    if not isinstance(message, dict):
        return None

    inner = message.get("message")
    msg = inner if isinstance(inner, dict) else message
    if msg.get("role") != "assistant":
        return None

    stop = msg.get("stopReason")
    if stop not in ("error", "aborted", "length"):
        return None

    err = msg.get("errorMessage")
    if isinstance(err, str) and err.strip():
        return err.strip()
    if err is not None:
        return str(err)
    if stop == "aborted":
        return _ABORTED_MSG
    if stop == "length":
        return _LENGTH_MSG
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

        # Start the decode clock at the first streamed assistant delta, not at
        # request start. This excludes prompt prefill/TTFT and tool execution.
        if ame_type in {
            "text_start",
            "text_delta",
            "thinking_start",
            "thinking_delta",
            "toolcall_start",
            "toolcall_delta",
        } and state.generation_started_at is None:
            state.generation_started_at = time.monotonic()

        if ame_type == "text_delta":
            delta = ame.get("delta", "") or ame.get("text", "")
            if delta:
                state.text += delta
                return ("text", delta)

        if ame_type == "toolcall_start":
            # Count the call here, but do not emit progress yet. Pi follows this
            # event with tool_execution_start, which has the authoritative tool
            # name and complete arguments. Emitting both events duplicated every
            # call, and the shared progress throttler then discarded most of the
            # useful execution events while surfacing placeholders like
            # "[tool:?]" instead.
            state.tool_count += 1
            return None

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
            # Pi may auto-retry this attempt after agent_end. Record the
            # failure now, but only surface it if retries are exhausted.
            state.saw_error = True
            state.terminal_error = str(error or "Pi error")
            return None

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
            full_pieces: list[str] = []

            exit_code = event.get("exitCode")
            if exit_code is not None:
                status = f"exit={exit_code}"
                pieces.append(status)
                full_pieces.append(status)

            if isinstance(result_content, list):
                for part in result_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = str(part.get("text", "")).strip()
                        if text:
                            preview = text[:177] + "..." if len(text) > 180 else text
                            pieces.append(preview)
                            full_pieces.append(text)
                            break

            is_error = event.get("isError", False)
            if is_error:
                pieces.append("ERROR")
                full_pieces.append("ERROR")

            suffix = f" {' | '.join(pieces)}" if pieces else ""
            full_suffix = f" {' | '.join(full_pieces)}" if full_pieces else ""
            desc = f"[tool-result:{name}{suffix}]"
            full_desc = f"[tool-result:{name}{full_suffix}]"
            self._log_to_file(f"{desc}\n")
            return (
                "tool_result",
                ToolResult(desc, full_desc if full_desc != desc else None),
            )

        return None

    def _record_terminal_error(self, error: str, state: RunState) -> Event:
        state.saw_error = True
        state.terminal_error = error
        self._log_to_file(f"[error] {error}\n")
        # Skip abort/length (expected). Process crashes already log in PiRunner.
        if error not in _NOISE_TERMINAL_ERRORS:
            log.error("Pi terminal error: %s", error)
        return ("error", error)

    def _handle_agent_end(self, event: dict, state: RunState) -> list[Event]:
        # agent_end is a step boundary. Pi may still auto-retry or compact and
        # continue afterward, so defer all terminal reporting to agent_settled.
        error = _terminal_error_from_messages(event.get("messages"))
        if error:
            state.saw_error = True
            state.terminal_error = error
        return []

    def _handle_agent_settled(self, state: RunState) -> list[Event]:
        state.saw_result = True
        if self._log_response and state.text:
            self._log_response(state.text)
        if state.saw_error and state.terminal_error:
            return [self._record_terminal_error(state.terminal_error, state)]
        return []

    def _handle_message_end(self, event: dict, state: RunState) -> list[Event]:
        events: list[Event] = []
        message = event.get("message")
        if isinstance(message, dict) and message.get("role") == "assistant":
            usage = message.get("usage")
            if isinstance(usage, dict):
                state.tokens_in += int(usage.get("input", 0) or 0)
                output = int(usage.get("output", 0) or 0)
                state.tokens_out += output
                state.tokens_cache_read += int(usage.get("cacheRead", 0) or 0)
                state.tokens_cache_write += int(usage.get("cacheWrite", 0) or 0)
                state.generation_tokens += output
                cost = usage.get("cost")
                if isinstance(cost, dict):
                    state.cost += float(cost.get("total", 0) or 0)

            if state.generation_started_at is not None:
                state.generation_duration_s += max(
                    0.0, time.monotonic() - state.generation_started_at
                )
                state.generation_started_at = None

        error = _assistant_terminal_error(message)
        if error:
            state.saw_error = True
            state.terminal_error = error
        elif isinstance(message, dict) and message.get("role") == "assistant":
            # A successful retry/continuation supersedes an earlier failed or
            # length-limited attempt in the same Pi run.
            state.saw_error = False
            state.terminal_error = None
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
        tokens_in = int(tokens.get("input", 0) or usage.get("input", 0) or state.tokens_in)
        tokens_out = int(tokens.get("output", 0) or usage.get("output", 0) or state.tokens_out)
        tokens_cache_read = int(
            tokens.get("cacheRead", 0)
            or usage.get("cacheRead", 0)
            or state.tokens_cache_read
        )
        tokens_cache_write = int(
            tokens.get("cacheWrite", 0)
            or usage.get("cacheWrite", 0)
            or state.tokens_cache_write
        )
        tokens_total = int(tokens.get("total", 0) or usage.get("total", 0) or 0)
        if tokens_total <= 0:
            tokens_total = (
                tokens_in + tokens_out + tokens_cache_read + tokens_cache_write
            )

        cost_info = usage.get("cost", {})
        cost_usd = 0.0
        if isinstance(cost_info, dict):
            for v in cost_info.values():
                if isinstance(v, (int, float)):
                    cost_usd += float(v)
        elif isinstance(cost_info, (int, float)):
            cost_usd = float(cost_info)
        if cost_usd <= 0:
            cost_usd = float(state.cost)

        model = str(usage.get("model", "pi") or "pi")
        context_usage = usage.get("contextUsage")
        if not isinstance(context_usage, dict):
            context_usage = {}

        result = {
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
            "generation_duration_s": float(state.generation_duration_s),
            "text": state.text,
            "summary": (
                f"[pi {tokens_in}/{tokens_out} tok"
                f" c{tokens_cache_read}/{tokens_cache_write}"
                f" ${cost_usd:.3f} {state.duration_s:.1f}s]"
            ),
        }
        for key in ("session_tokens_total", "session_cost_total"):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                result[key] = value
        context_fields = {
            "context_tokens": context_usage.get("tokens"),
            "context_window": context_usage.get("contextWindow"),
            "context_percent": context_usage.get("percent"),
        }
        result.update({k: v for k, v in context_fields.items() if v is not None})
        return result

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

        if event_type == "agent_settled":
            return self._handle_agent_settled(state)

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
