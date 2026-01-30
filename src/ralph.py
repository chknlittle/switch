"""Ralph helpers.

Ralph is implemented as a mode of SessionRuntime.
This module only keeps the command parser.
"""

from __future__ import annotations

import shlex


def parse_ralph_command(body: str) -> dict | None:
    """Parse /ralph command into components.

    Formats:
        /ralph <prompt> --max <N> --done "<promise>" --wait <M>
        /ralph <N> <prompt>  (shorthand)
        /ralph <prompt>  (infinite - dangerous!)

    Returns dict with: prompt, max_iterations, completion_promise, wait_minutes
    """
    if not body.lower().startswith("/ralph"):
        return None

    rest = body[6:].strip()
    if not rest:
        return None

    try:
        parts = shlex.split(rest)
    except ValueError:
        parts = rest.split()

    max_iterations = 0
    completion_promise = None
    wait_minutes = 2.0 / 60.0
    prompt_parts = []

    i = 0
    while i < len(parts):
        part = parts[i]
        if part in ("--max", "--max-iterations", "-m") and i + 1 < len(parts):
            try:
                max_iterations = int(parts[i + 1])
                i += 2
                continue
            except ValueError:
                pass
        elif part in ("--done", "--completion-promise", "-d") and i + 1 < len(parts):
            completion_promise = parts[i + 1]
            i += 2
            continue
        elif part in ("--wait", "--wait-min", "--wait-minutes", "--interval", "--sleep", "-w") and i + 1 < len(parts):
            try:
                wait_minutes = float(parts[i + 1])
                i += 2
                continue
            except ValueError:
                pass
        prompt_parts.append(part)
        i += 1

    # Shorthand: first number is max iterations
    if prompt_parts and prompt_parts[0].isdigit():
        max_iterations = int(prompt_parts[0])
        prompt_parts = prompt_parts[1:]

    prompt = " ".join(prompt_parts)
    if not prompt:
        return None
    return {
        "prompt": prompt,
        "max_iterations": max_iterations,
        "completion_promise": completion_promise,
        "wait_minutes": max(0.0, wait_minutes),
    }
