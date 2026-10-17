"""Projects: broad labels backed by memory folders.

A project is any folder in the memory vault containing an empty `.project`
marker, e.g. `memory/lemebase/.project`. Messages to a dispatcher that start
with `#<name>` create a session tagged with that project.
"""

from __future__ import annotations

import re
from pathlib import Path

MEMORY_DIR = Path(__file__).resolve().parent.parent / "memory"
PROJECT_MARKER = ".project"

# One word only: keeps projects broad.
_PREFIX_RE = re.compile(r"^#([a-z0-9-]+)(?:\s+|$)", re.IGNORECASE)


def list_projects(memory_dir: Path = MEMORY_DIR) -> list[str]:
    if not memory_dir.is_dir():
        return []
    return sorted(
        p.name for p in memory_dir.iterdir() if (p / PROJECT_MARKER).is_file()
    )


def parse_project_prefix(body: str) -> tuple[str, str] | None:
    """Split `#name rest` into (name, rest). None if there is no `#name` prefix."""
    m = _PREFIX_RE.match(body)
    if not m:
        return None
    return m.group(1).lower(), body[m.end() :].strip()


def project_context(name: str, memory_dir: Path = MEMORY_DIR) -> str:
    folder = memory_dir / name
    return (
        "[Switch project context]\n"
        f"This session belongs to project #{name}. "
        f"Its memory folder is {folder}/. Before starting, read what is there "
        f"(ls + grep -r) for relevant context. Save learnings from this session "
        f"into that folder, not a different topic folder."
    )
