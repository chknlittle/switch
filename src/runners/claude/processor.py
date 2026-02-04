"""Claude runner event processing.

Separates parsing/logging concerns from the subprocess orchestration in
`src/runners/claude.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.runners.base import RunState
from src.runners.ports import RunnerEvent
from src.runners.tool_logging import (
    format_tool_input_preview,
    should_log_tool_input,
    tool_input_max_len,
)


@dataclass
class ClaudeResult:
    """Final result from a Claude run."""

    text: str
    session_id: str | None
    cost: float
    turns: int
    tool_count: int
    total_tokens: int
    context_window: int
    duration_s: float


Event = RunnerEvent


class ClaudeEventProcessor:
    def __init__(
        self,
        *,
        log_to_file: Callable[[str], None],
        log_response: Callable[[str], None],
    ):
        self._log_to_file = log_to_file
        self._log_response = log_response

    def _handle_system_init(self, event: dict, state: RunState) -> Event | None:
        if event.get("subtype") != "init":
            return None
        session_id = event.get("session_id")
        if session_id:
            state.session_id = session_id
            return ("session_id", session_id)
        return None

    def _handle_assistant_text(self, block: dict, state: RunState) -> Event | None:
        text = block.get("text", "").strip()
        if not text:
            return None
        state.text = text
        self._log_response(text)
        return ("text", text)

    def _handle_assistant_tool(self, block: dict, state: RunState) -> Event | None:
        state.tool_count += 1
        name = block.get("name", "?")
        tool_input = block.get("input", {})

        tool_id = str(name).strip().lower() if name else "?"

        def _clean_label(value: object, *, max_len: int = 180) -> str | None:
            if not isinstance(value, str):
                return None
            s = " ".join(value.split())
            if not s:
                return None
            if len(s) > max_len:
                return s[: max_len - 3] + "..."
            return s

        title = (
            _clean_label(tool_input.get("title"))
            if isinstance(tool_input, dict)
            else None
        )
        description = (
            _clean_label(tool_input.get("description"))
            if isinstance(tool_input, dict)
            else None
        )
        if title and description and title == description:
            description = None

        extra_bits: list[str] = []
        if title:
            extra_bits.append(title)
        if description:
            extra_bits.append(description)

        if name == "Bash":
            preview = ""
            if isinstance(tool_input, dict):
                preview = str(tool_input.get("command", "") or "")
            preview = _clean_label(preview, max_len=80) or ""
            if preview:
                extra_bits = [preview] + extra_bits
            extra = " | ".join(extra_bits)
            desc = f"[tool:{tool_id} {extra}]" if extra else f"[tool:{tool_id}]"
        elif name in ("Read", "Write", "Edit"):
            path = ""
            if isinstance(tool_input, dict):
                path = str(tool_input.get("file_path", "") or "")
            leaf = Path(path).name if path else ""
            if leaf:
                extra_bits = [leaf] + extra_bits
            extra = " | ".join(extra_bits)
            desc = f"[tool:{tool_id} {extra}]" if extra else f"[tool:{tool_id}]"
        else:
            extra = " | ".join(extra_bits)
            desc = (
                f"[tool:{tool_id} {extra}]"
                if (tool_id and extra)
                else (f"[tool:{tool_id}]" if tool_id else "[tool:?]")
            )

        if should_log_tool_input():
            formatted = format_tool_input_preview(tool_id, tool_input)
            if formatted:
                formatted = formatted[: tool_input_max_len()]
                self._log_to_file(f"{desc}\n  input: {formatted}\n")
                return ("tool", f"{desc} input: {formatted}")

        self._log_to_file(f"{desc}\n")
        return ("tool", desc)

    def _handle_assistant(self, event: dict, state: RunState) -> list[Event]:
        events: list[Event] = []
        content = event.get("message", {}).get("content", [])

        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                result = self._handle_assistant_text(block, state)
                if result:
                    events.append(result)
            elif block_type == "tool_use":
                result = self._handle_assistant_tool(block, state)
                if result:
                    events.append(result)

        return events

    def _handle_result(self, event: dict, state: RunState) -> Event:
        state.saw_result = True

        if event.get("is_error"):
            state.saw_error = True
            return ("error", event.get("result", "Unknown error"))

        cost = event.get("total_cost_usd", 0)
        turns = event.get("num_turns", 0)
        duration = event.get("duration_ms", 0) / 1000

        usage = event.get("usage", {})
        total_tokens = (
            usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
            + usage.get("output_tokens", 0)
        )

        context_window = 200000
        model_name = "claude"
        for name, model_data in event.get("modelUsage", {}).items():
            model_name = name
            context_window = model_data.get("contextWindow", context_window)
            break

        tokens_k = total_tokens / 1000
        context_k = context_window / 1000
        summary = (
            f"[{model_name} {turns}t {state.tool_count}tools ${cost:.3f} {duration:.1f}s"
            f" | {tokens_k:.1f}k/{context_k:.0f}k]"
        )

        payload = {
            "engine": "claude",
            "model": model_name,
            "turns": turns,
            "tool_count": state.tool_count,
            "tokens_total": total_tokens,
            "context_window": context_window,
            "cost_usd": float(cost),
            "duration_s": float(duration),
            "text": state.text,
            "summary": summary,
        }

        return ("result", payload)

    def parse_event(self, event: dict, state: RunState) -> list[Event]:
        event_type = event.get("type")

        if event_type == "system":
            result = self._handle_system_init(event, state)
            return [result] if result else []
        if event_type == "assistant":
            return self._handle_assistant(event, state)
        if event_type == "result":
            return [self._handle_result(event, state)]

        return []
