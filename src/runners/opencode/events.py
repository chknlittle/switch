"""OpenCode event normalization helpers."""

from __future__ import annotations


def _find_session_id_in_dict(d: dict) -> str | None:
    """Check a dict for sessionID/sessionId/session_id keys."""
    for key in ("sessionID", "sessionId", "session_id"):
        value = d.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def extract_session_id(payload: dict) -> str | None:
    """Extract session ID from various event structures.

    OpenCode GlobalBus wraps events as:
    {"directory": "...", "payload": {"type": "...", "properties": {"sessionID": "..."}}}

    For part events, sessionID is inside the part object:
    {"payload": {"properties": {"part": {"sessionID": "..."}}}}

    We check multiple nesting levels to handle all cases.
    """
    # Check top level
    if result := _find_session_id_in_dict(payload):
        return result

    # Check payload.properties (direct event structure)
    props = payload.get("properties")
    if isinstance(props, dict):
        if result := _find_session_id_in_dict(props):
            return result
        # Check payload.properties.part (part events)
        part = props.get("part")
        if isinstance(part, dict):
            if result := _find_session_id_in_dict(part):
                return result

    # Check payload.payload.properties (GlobalBus wrapped structure)
    inner_payload = payload.get("payload")
    if isinstance(inner_payload, dict):
        inner_props = inner_payload.get("properties")
        if isinstance(inner_props, dict):
            if result := _find_session_id_in_dict(inner_props):
                return result
            # Check payload.payload.properties.part (wrapped part events)
            inner_part = inner_props.get("part")
            if isinstance(inner_part, dict):
                if result := _find_session_id_in_dict(inner_part):
                    return result

    return None


def coerce_event(payload: dict) -> dict | None:
    # OpenCode GlobalBus SSE wraps events as:
    # {"directory": "...", "payload": {"type": "...", "properties": {...}}}
    # Unwrap so downstream normalization works regardless of endpoint.
    inner = payload.get("payload")
    if isinstance(inner, dict) and isinstance(inner.get("type"), str):
        payload = inner

    if "type" in payload and "part" in payload:
        return payload

    event_type = payload.get("type")
    if not isinstance(event_type, str):
        return None

    props = (
        payload.get("properties")
        if isinstance(payload.get("properties"), dict)
        else None
    )

    if props:
        # OpenCode server mode tends to emit message events instead of raw "text"
        # events. Normalize them into the minimal shapes the runner expects.
        if event_type == "message.updated":
            info = props.get("info")
            if isinstance(info, dict):
                return {
                    "type": "message_meta",
                    "sessionID": info.get("sessionID"),
                    "messageID": info.get("id"),
                    "role": info.get("role"),
                }

        if event_type == "message.part.updated":
            part = props.get("part")
            if isinstance(part, dict) and part.get("type") == "text":
                return {
                    "type": "message_part",
                    "sessionID": part.get("sessionID"),
                    "messageID": part.get("messageID"),
                    "text": part.get("text", ""),
                }

        if event_type in {"question.asked", "question"}:
            return {"type": "question.asked", **props}
        if event_type in {"permission.requested", "session.permission.requested"}:
            return {"type": "permission.requested", **props}
        if "part" in props and isinstance(props["part"], dict):
            part = props["part"]
            part_type = part.get("type")
            if part_type == "text":
                return {"type": "text", "part": {"text": part.get("text", "")}}
            if part_type in {"tool", "tool_use"}:
                return {"type": "tool_use", "part": part}
            if part_type in {
                "tool_result",
                "tool-result",
                "tool.output",
                "tool_output",
                "tool.response",
                "tool_response",
            }:
                return {"type": "tool_result", "part": part}
            if part_type in {"question", "question.asked"}:
                merged = {"type": "question.asked"}
                merged.update(part)
                return merged
        if event_type in {"error", "session.error"}:
            return {"type": "error", **props}

    if event_type in {
        "step_start",
        "step_finish",
        "text",
        "tool_use",
        "tool_result",
        "tool-result",
        "error",
    }:
        if event_type == "tool-result":
            payload = dict(payload)
            payload["type"] = "tool_result"
        return payload

    return None
