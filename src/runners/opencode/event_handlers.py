"""Per-event-type handlers for OpenCode SSE payloads."""

from __future__ import annotations

from typing import Callable

from src.runners.base import RunState
from src.runners.opencode.models import Event, PermissionRequest, Question
from src.runners.opencode.text_sanitize import sanitize_assistant_text
from src.runners.opencode.tool_format import (
    extract_desc_parts,
    extract_task_ref,
    extract_tool_input,
    format_tool_header,
    format_tool_result_suffix,
)
from src.runners.tool_logging import (
    format_tool_input_preview,
    should_log_tool_input,
    tool_input_max_len,
)


def apply_text_update(text: str, state: RunState) -> Event | None:
    if not text:
        return None

    state.raw_text = text
    text = sanitize_assistant_text(text)

    # SSE sends full accumulated text, not deltas - extract only the new part.
    if text.startswith(state.text):
        delta = text[len(state.text) :]
        state.text = text
        if delta:
            return ("text", delta)
        return None

    state.text = text
    return ("text", text)


def handle_step_start(event: dict, state: RunState) -> Event | None:
    session_id = event.get("sessionID")
    if isinstance(session_id, str) and session_id:
        state.session_id = session_id
        return ("session_id", session_id)
    return None


def handle_text(event: dict, state: RunState) -> Event | None:
    part = event.get("part", {})
    text = part.get("text", "") if isinstance(part, dict) else ""
    if isinstance(text, str):
        return apply_text_update(text, state)
    return None


def handle_message_meta(event: dict, state: RunState) -> Event | None:
    message_id = event.get("messageID")
    role = event.get("role")
    if (
        isinstance(message_id, str)
        and message_id
        and isinstance(role, str)
        and role
    ):
        state.message_roles[message_id] = role
    return None


def handle_message_part_meta(event: dict, state: RunState) -> Event | None:
    part_id = event.get("partID")
    part_type = event.get("partType")
    if (
        isinstance(part_id, str)
        and part_id
        and isinstance(part_type, str)
        and part_type
    ):
        state.message_part_types[part_id] = part_type
    return None


def handle_message_part_delta(event: dict, state: RunState) -> Event | None:
    message_id = event.get("messageID")
    role = (
        state.message_roles.get(message_id)
        if isinstance(message_id, str) and message_id
        else None
    )
    if role != "assistant":
        return None

    part_id = event.get("partID")
    part_type = (
        state.message_part_types.get(part_id)
        if isinstance(part_id, str) and part_id
        else None
    )
    if part_type is not None and part_type != "text":
        return None

    text = event.get("text", "")
    if not isinstance(text, str) or not text:
        return None

    state.raw_text += text
    visible_text = sanitize_assistant_text(state.raw_text)
    if not visible_text.startswith(state.text):
        state.text = visible_text
        return ("text", visible_text)

    delta = visible_text[len(state.text) :]
    state.text = visible_text
    if not delta:
        return None
    return ("text", delta)


def handle_tool_use(
    event: dict,
    state: RunState,
    *,
    log_to_file: Callable[[str], None],
) -> Event | None:
    part = event.get("part", {})
    if not isinstance(part, dict):
        return None

    tool = part.get("tool")
    if not tool:
        return None

    tool_str = str(tool)

    # Deduplicate: SSE can send multiple updates for the same tool call.
    # Tool input/args often arrives on a later update, so allow a follow-up
    # event to log input even if we already logged the tool header.
    tool_id = part.get("id") or part.get("toolUseId") or part.get("callID")
    if not tool_id:
        msg_id = part.get("messageID", "")
        idx = part.get("index", "")
        if msg_id or idx:
            tool_id = f"{msg_id}:{idx}"

    tool_state = part.get("state", {})
    tool_input = extract_tool_input(part, tool_state, tool_str)
    has_input = tool_input is not None

    title, description = extract_desc_parts(
        tool=tool_str,
        part_obj=part,
        tool_state_obj=tool_state,
        tool_input_obj=tool_input,
    )

    task_ref = extract_task_ref(part, tool_state, tool_str)
    desc, has_rich_header = format_tool_header(
        tool_str,
        title=title,
        description=description,
        task_ref=task_ref,
    )

    if tool_id and tool_id in state.seen_tool_ids:
        if (
            has_input
            and should_log_tool_input()
            and tool_id not in state.tool_input_logged_ids
        ):
            formatted = format_tool_input_preview(tool_str, tool_input)
            if formatted:
                max_len = tool_input_max_len()
                formatted = formatted[:max_len]
                log_to_file(f"  input: {formatted}\n")
                state.tool_input_logged_ids.add(tool_id)
                return ("tool", f"[tool:{tool}] input: {formatted}")

        # If a follow-up SSE update finally contains useful title/command
        # info, emit one upgraded header even when input logging is off.
        if has_rich_header and tool_id not in state.tool_header_upgraded_ids:
            log_to_file(f"{desc}\n")
            state.tool_header_upgraded_ids.add(tool_id)
            return ("tool", desc)
        return None

    if tool_id:
        state.seen_tool_ids.add(tool_id)

    state.tool_count += 1
    if tool_id and has_rich_header:
        state.tool_header_upgraded_ids.add(tool_id)

    if should_log_tool_input():
        formatted = format_tool_input_preview(tool_str, tool_input)
        if formatted:
            max_len = tool_input_max_len()
            formatted = formatted[:max_len]
            log_to_file(f"{desc}\n  input: {formatted}\n")
            if tool_id:
                state.tool_input_logged_ids.add(tool_id)
            return ("tool", f"{desc} input: {formatted}")

    log_to_file(f"{desc}\n")
    return ("tool", desc)


def handle_tool_result(
    event: dict,
    state: RunState,
    *,
    log_to_file: Callable[[str], None],
) -> Event | None:
    part = event.get("part", {})
    if not isinstance(part, dict):
        return None

    tool = part.get("tool") or part.get("name") or "tool"
    tool_str = str(tool)

    tool_id = part.get("id") or part.get("toolUseId") or part.get("callID")
    if not tool_id:
        msg_id = part.get("messageID", "")
        idx = part.get("index", "")
        if msg_id or idx:
            tool_id = f"{msg_id}:{idx}:result"

    if tool_id and tool_id in state.tool_result_seen_ids:
        return None
    if tool_id:
        state.tool_result_seen_ids.add(tool_id)

    state_obj = part.get("state")
    suffix = format_tool_result_suffix(tool_str, part=part, state_obj=state_obj)
    desc = f"[tool-result:{tool_str}{suffix}]"
    log_to_file(f"{desc}\n")
    return ("tool_result", desc)


def handle_step_finish(
    event: dict,
    state: RunState,
    *,
    make_result: Callable[[RunState], dict],
) -> Event | None:
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
        return ("result", make_result(state))
    return None


def handle_error(event: dict, state: RunState) -> Event:
    state.saw_error = True
    message = event.get("message")
    error = event.get("error")

    if isinstance(message, dict):
        message = message.get("data", {}).get("message") or message.get("message")

    return ("error", str(message or error or "OpenCode error"))


def handle_question(
    event: dict,
    state: RunState,
    *,
    log_to_file: Callable[[str], None],
) -> Event | None:
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
        log_to_file(f"Question event missing request ID: {event}\n")
        return None

    question = Question(request_id=request_id, questions=questions)
    log_to_file(f"\n[QUESTION] {request_id}: {questions}\n")
    return ("question", question)


def handle_permission_request(
    event: dict,
    state: RunState,
    *,
    log_to_file: Callable[[str], None],
) -> Event | None:
    request_id = event.get("requestID") or event.get("id")
    permission = event.get("permission")
    patterns = event.get("patterns")
    message = event.get("message")

    if not isinstance(request_id, str) or not request_id:
        log_to_file(f"Permission event missing request ID: {event}\n")
        return None
    if not isinstance(permission, str) or not permission:
        permission = "permission"

    normalized_patterns: list[str] = []
    if isinstance(patterns, list):
        normalized_patterns = [p for p in patterns if isinstance(p, str) and p]

    request = PermissionRequest(
        request_id=request_id,
        permission=permission,
        patterns=normalized_patterns,
        message=message if isinstance(message, str) and message else None,
    )
    log_to_file(
        f"\n[PERMISSION] {request_id}: {permission} {normalized_patterns}\n"
    )
    return ("permission", request)


def handle_message_part(event: dict, state: RunState) -> Event | None:
    part_id = event.get("partID")
    if isinstance(part_id, str) and part_id:
        state.message_part_types[part_id] = "text"

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
        return apply_text_update(text, state)
    return None
