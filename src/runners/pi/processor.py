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
            error = ame.get("error", "Pi error")
            state.saw_error = True
            return ("error", str(error))

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

    def _handle_agent_end(self, event: dict, state: RunState) -> Event | None:
        state.saw_result = True

        if self._log_response and state.text:
            self._log_response(state.text)

        # Stats come from get_session_stats — we'll inject them from the runner.
        # For now, build a minimal result payload.
        return None  # Runner handles stats separately.

    def make_result(self, state: RunState, stats: dict | None = None) -> dict:
        if self._log_response and state.text:
            self._log_response(state.text)

        usage = stats or {}
        tokens_in = int(usage.get("input", 0) or 0)
        tokens_out = int(usage.get("output", 0) or 0)
        tokens_cache_read = int(usage.get("cacheRead", 0) or 0)
        tokens_cache_write = int(usage.get("cacheWrite", 0) or 0)
        tokens_total = int(usage.get("total", 0) or 0)

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
            self._handle_agent_end(event, state)
            # Result is emitted by the runner after fetching stats.
            return []

        if event_type == "extension_ui_request":
            # TODO: map to ("question", Question) for interactive prompts.
            pass

        return []
