"""Search the knowledge graph."""

from __future__ import annotations

from typing import Any

from src.runners.weaver.graph import GraphClient


async def graph_query(
    client: GraphClient,
    query: str,
    *,
    group_id: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    return await client.search(query, group_id=group_id, limit=limit)
