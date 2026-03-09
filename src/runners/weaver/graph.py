"""Graphiti knowledge graph client lifecycle."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_RRF,
)

from src.runners.weaver.config import WeaverConfig

log = logging.getLogger(__name__)


class GraphClient:
    """Manages a Graphiti instance connected to FalkorDB."""

    def __init__(self, config: WeaverConfig):
        self.config = config
        self._graphiti: Graphiti | None = None

    async def init(self) -> None:
        llm_config = LLMConfig(
            api_key="local",
            base_url=self.config.llm_base_url,
            model=self.config.llm_model,
            small_model=self.config.llm_model,
        )
        llm_client = OpenAIClient(config=llm_config)
        self._graphiti = Graphiti(
            self.config.falkordb_uri,
            llm_client=llm_client,
        )
        try:
            await self._graphiti.build_indices()
        except Exception:
            log.warning("graph: could not build indices (may already exist)", exc_info=True)

    async def close(self) -> None:
        if self._graphiti:
            try:
                await self._graphiti.close()
            except Exception:
                log.debug("graph: close error", exc_info=True)
            self._graphiti = None

    async def search(self, query: str, *, group_id: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        if not self._graphiti:
            return []
        try:
            config = COMBINED_HYBRID_SEARCH_RRF
            config.limit = limit
            results = await self._graphiti.search_(
                query=query,
                config=config,
                group_ids=[group_id] if group_id else None,
            )
            items = []
            for node in (results.nodes or []):
                items.append({
                    "type": "entity",
                    "name": getattr(node, "name", str(node)),
                    "summary": getattr(node, "summary", ""),
                })
            for edge in (results.edges or []):
                items.append({
                    "type": "relation",
                    "name": getattr(edge, "name", str(edge)),
                    "fact": getattr(edge, "fact", ""),
                })
            return items
        except Exception:
            log.warning("graph: search failed", exc_info=True)
            return []

    async def add_episode(
        self,
        content: str,
        *,
        group_id: str | None = None,
        source: str = "message",
    ) -> dict[str, Any]:
        if not self._graphiti:
            return {"status": "graph not initialized"}
        try:
            result = await self._graphiti.add_episode(
                name=f"weaver-{source}",
                episode_body=content,
                source_description=f"weaver {source}",
                reference_time=datetime.now(timezone.utc),
                group_id=group_id or "default",
            )
            return {
                "status": "indexed",
                "entities": len(getattr(result, "nodes", []) if result else []),
                "relations": len(getattr(result, "edges", []) if result else []),
            }
        except Exception:
            log.warning("graph: add_episode failed", exc_info=True)
            return {"status": "error"}
