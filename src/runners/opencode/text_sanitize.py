"""Assistant text sanitization for OpenCode streams."""

from __future__ import annotations

import re


def sanitize_assistant_text(text: str) -> str:
    if not text:
        return ""

    # Some local models leak internal reasoning inside <think> blocks, or
    # emit an orphaned closing tag with the visible answer after it.
    text = re.sub(
        r"<think>.*?</think>\s*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.lstrip()
