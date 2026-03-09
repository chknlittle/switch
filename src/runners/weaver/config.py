"""Weaver runner configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class WeaverConfig:
    llm_base_url: str = "http://localhost:8080/v1"
    llm_model: str = "qwen3.5-122b"
    falkordb_uri: str = "bolt://localhost:6379"
    searxng_url: str = "http://localhost:8888"
    max_tool_rounds: int = 15
    graph_search_results: int = 10

    @staticmethod
    def from_env() -> WeaverConfig:
        return WeaverConfig(
            llm_base_url=os.getenv("WEAVER_LLM_URL", "http://localhost:8080/v1"),
            llm_model=os.getenv("WEAVER_LLM_MODEL", "qwen3.5-122b"),
            falkordb_uri=os.getenv("WEAVER_FALKORDB_URI", "bolt://localhost:6379"),
            searxng_url=os.getenv("WEAVER_SEARXNG_URL", "http://localhost:8888"),
        )
