"""Tool-use and tool-result formatting helpers for OpenCode events."""

from __future__ import annotations

from src.runners.tool_logging import format_tool_input_preview


def clean_label(value: object, *, max_len: int = 180) -> str | None:
    if not isinstance(value, str):
        return None
    s = " ".join(value.split())
    if not s:
        return None
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def extract_tool_input(
    part_obj: dict, tool_state_obj: object, tool: str
) -> object | None:
    raw_input: object | None = None
    if isinstance(tool_state_obj, dict):
        for key in ("input", "args", "arguments", "params"):
            value = tool_state_obj.get(key)
            if value is not None:
                raw_input = value
                break

        # Some server builds expose bash command directly on state.
        if raw_input is None and tool == "bash":
            cmd = tool_state_obj.get("command")
            if isinstance(cmd, str) and cmd.strip():
                raw_input = {"command": cmd.strip()}
    if raw_input is None:
        for key in ("input", "args", "arguments", "params"):
            value = part_obj.get(key)
            if value is not None:
                raw_input = value
                break

    if raw_input is None and tool == "bash":
        cmd = part_obj.get("command")
        if isinstance(cmd, str) and cmd.strip():
            raw_input = {"command": cmd.strip()}
    return raw_input


def _is_meaningful_preview(value: str | None) -> bool:
    if not value:
        return False
    v = value.strip()
    if not v:
        return False
    return v not in {"{}", "[]", '""', "null", "None"}


def extract_desc_parts(
    *,
    tool: str,
    part_obj: dict,
    tool_state_obj: object,
    tool_input_obj: object | None,
) -> tuple[str | None, str | None]:
    title = None
    description = None

    if isinstance(tool_state_obj, dict):
        title = clean_label(tool_state_obj.get("title"))
        description = clean_label(tool_state_obj.get("description"))

    if description is None and isinstance(part_obj, dict):
        description = clean_label(part_obj.get("description"))

    # Tool schemas commonly include a per-call description inside args/input.
    if isinstance(tool_input_obj, dict):
        if title is None:
            title = clean_label(tool_input_obj.get("title"))
        if description is None:
            description = clean_label(tool_input_obj.get("description"))

        # Keep bash progress readable even when full tool-input logging is
        # disabled. Show a short command preview in the tool header.
        if tool == "bash" and title is None:
            cmd = tool_input_obj.get("command")
            title = clean_label(cmd, max_len=100)

    # Some servers send bash input as a plain string command.
    if tool == "bash" and title is None and isinstance(tool_input_obj, str):
        title = clean_label(tool_input_obj, max_len=100)

    # Generic fallback: show a compact preview when available so tool
    # progress stays informative even when input logging is disabled.
    if title is None and description is None and tool_input_obj is not None:
        preview = format_tool_input_preview(tool, tool_input_obj)
        if _is_meaningful_preview(preview):
            title = clean_label(preview, max_len=100)

    # Avoid duplicating identical strings.
    if title and description and title == description:
        description = None

    return title, description


def extract_task_ref(part_obj: dict, tool_state_obj: object, tool: str) -> str | None:
    if tool != "task":
        return None

    candidates: list[object] = [tool_state_obj, part_obj]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        for key in ("task_id", "taskId"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return f"task {value.strip()}"

        metadata = candidate.get("metadata")
        if isinstance(metadata, dict):
            for key in ("sessionId", "sessionID", "task_id", "taskId"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return f"child {value.strip()}"

    return None


def pick(obj: object, keys: tuple[str, ...]) -> object | None:
    if not isinstance(obj, dict):
        return None
    for key in keys:
        value = obj.get(key)
        if value is not None:
            return value
    return None


def format_tool_header(
    tool: str,
    *,
    title: str | None,
    description: str | None,
    task_ref: str | None,
) -> tuple[str, bool]:
    extra_bits: list[str] = []
    if title:
        extra_bits.append(title)
    if description:
        extra_bits.append(description)
    if task_ref:
        extra_bits.append(task_ref)
    extra = " | ".join(extra_bits)
    desc = f"[tool:{tool} {extra}]" if extra else f"[tool:{tool}]"
    return desc, bool(extra)


def format_tool_result_suffix(
    tool_str: str,
    *,
    part: dict,
    state_obj: object,
) -> str:
    exit_code = pick(part, ("exitCode", "exit_code", "code"))
    if exit_code is None:
        exit_code = pick(state_obj, ("exitCode", "exit_code", "code"))

    output = pick(part, ("output", "stdout", "stderr", "result", "text"))
    if output is None:
        output = pick(
            state_obj,
            (
                "output",
                "stdout",
                "stderr",
                "result",
                "response",
                "text",
                "error",
            ),
        )

    pieces: list[str] = []

    if tool_str == "task":
        task_ref = None
        if isinstance(state_obj, dict):
            metadata = state_obj.get("metadata")
            if isinstance(metadata, dict):
                for key in ("sessionId", "sessionID", "task_id", "taskId"):
                    value = metadata.get(key)
                    if isinstance(value, str) and value.strip():
                        task_ref = value.strip()
                        break
        if task_ref is None:
            for key in ("task_id", "taskId"):
                value = pick(part, (key,)) or pick(state_obj, (key,))
                if isinstance(value, str) and value.strip():
                    task_ref = value.strip()
                    break
        if task_ref:
            pieces.append(f"child={task_ref}")

    if exit_code is not None:
        pieces.append(f"exit={exit_code}")

    if isinstance(output, str):
        compact = " ".join(output.split())
        if compact:
            if len(compact) > 180:
                compact = compact[:177] + "..."
            pieces.append(compact)

    if not pieces:
        status = pick(part, ("status",)) or pick(state_obj, ("status",))
        if isinstance(status, str) and status.strip():
            pieces.append(status.strip())

    return f" {' | '.join(pieces)}" if pieces else ""
