"""SearXNG web search tool."""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

log = logging.getLogger(__name__)


async def web_search(query: str, *, searxng_url: str, max_results: int = 8) -> list[dict[str, str]]:
    url = f"{searxng_url}/search"
    params = {"q": query, "format": "json"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return [{"error": f"SearXNG returned status {resp.status}"}]
                data = await resp.json()
    except Exception as e:
        log.warning("web_search failed: %s", e)
        return [{"error": str(e)}]

    results = []
    for item in data.get("results", [])[:max_results]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content", ""),
        })
    return results
