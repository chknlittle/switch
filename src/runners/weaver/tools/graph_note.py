"""Explicitly save ideas/findings to the knowledge graph."""

from __future__ import annotations

from typing import Any

from src.runners.weaver.graph import GraphClient


async def graph_note(
    client: GraphClient,
    content: str,
    *,
    topic: str | None = None,
    group_id: str | None = None,
) -> dict[str, Any]:
    body = content
    if topic:
        body = f"[{topic}] {content}"
    return await client.add_episode(body, group_id=group_id, source="note")
