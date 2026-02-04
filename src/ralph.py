"""Ralph helpers.

Ralph is implemented as a mode of SessionRuntime.
This module only keeps the command parser.
"""

from __future__ import annotations

import shlex


_UNICODE_DASHES = {
    "\u2010",  # hyphen
    "\u2011",  # non-breaking hyphen
    "\u2012",  # figure dash
    "\u2013",  # en dash
    "\u2014",  # em dash
    "\u2212",  # minus sign
    "\ufe58",  # small em dash
    "\ufe63",  # small hyphen-minus
    "\uff0d",  # fullwidth hyphen-minus
}


def parse_ralph_command(body: str) -> dict | None:
    """Parse /ralph command into components.

    Formats:
        /ralph <prompt> --max <N> --done "<promise>" --wait <M> [--look]
        /ralph <N> <prompt>  (shorthand)
        /ralph <prompt>  (infinite - dangerous!)

    Returns dict with: prompt, max_iterations, completion_promise, wait_minutes, prompt_only, swarm
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

    # Be forgiving about unicode dashes. On phones/chat apps, users often type
    # "--wait" but it gets auto-replaced into an em dash ("—wait").
    parts = [
        ("--" + p[1:]) if (p and p[0] in _UNICODE_DASHES and not p.startswith("--")) else p
        for p in parts
    ]

    # Optional: /ralph ... --swarm N
    # This is primarily consumed by dispatcher/session command handlers which
    # fan out into multiple sessions. We strip it from the forwarded args so it
    # doesn't end up in the prompt.
    swarm = 1
    cleaned_parts: list[str] = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if part in ("--swarm", "--parallel") and i + 1 < len(parts):
            try:
                swarm = max(1, int(parts[i + 1]))
                i += 2
                continue
            except ValueError:
                pass
        elif part.startswith("--swarm=") or part.startswith("--parallel="):
            try:
                swarm = max(1, int(part.split("=", 1)[1]))
                i += 1
                continue
            except ValueError:
                pass
        cleaned_parts.append(part)
        i += 1

    # shlex.join preserves quoting; fallback keeps spaces sane.
    try:
        forward_args = shlex.join(cleaned_parts)
    except AttributeError:  # pragma: no cover (py<3.8)
        forward_args = " ".join(cleaned_parts)

    max_iterations = 0
    completion_promise = None
    wait_minutes = 2.0 / 60.0
    prompt_only = False
    prompt_parts = []

    i = 0
    while i < len(cleaned_parts):
        part = cleaned_parts[i]
        if part in ("--look", "--prompt-only", "--promptonly", "--stateless", "--isolated"):
            prompt_only = True
            i += 1
            continue
        if part in ("--max", "--max-iterations", "-m") and i + 1 < len(cleaned_parts):
            try:
                max_iterations = int(cleaned_parts[i + 1])
                i += 2
                continue
            except ValueError:
                pass
        elif part.startswith("--max="):
            try:
                max_iterations = int(part.split("=", 1)[1])
                i += 1
                continue
            except ValueError:
                pass
        elif part in ("--done", "--completion-promise", "-d") and i + 1 < len(cleaned_parts):
            completion_promise = cleaned_parts[i + 1]
            i += 2
            continue
        elif part.startswith("--done=") or part.startswith("--completion-promise="):
            completion_promise = part.split("=", 1)[1]
            i += 1
            continue
        elif part in ("--wait", "--wait-min", "--wait-minutes", "--interval", "--sleep", "-w") and i + 1 < len(cleaned_parts):
            try:
                wait_minutes = float(cleaned_parts[i + 1])
                i += 2
                continue
            except ValueError:
                pass
        elif part.startswith("--wait=") or part.startswith("--wait-min=") or part.startswith("--wait-minutes="):
            try:
                wait_minutes = float(part.split("=", 1)[1])
                i += 1
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
        "prompt_only": bool(prompt_only),
        "swarm": int(swarm or 1),
        "forward_args": forward_args,
    }
