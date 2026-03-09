"""Tool definitions (OpenAI function-calling format) and dispatch."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.runners.weaver.config import WeaverConfig
from src.runners.weaver.graph import GraphClient
from src.runners.weaver.tools.search import web_search
from src.runners.weaver.tools.read_url import read_url
from src.runners.weaver.tools.graph_query import graph_query
from src.runners.weaver.tools.graph_note import graph_note

log = logging.getLogger(__name__)

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using SearXNG. Returns titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": "Fetch and extract the text content of a web page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to read.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_query",
            "description": "Search the knowledge graph for relevant entities, facts, and relationships.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query about what to look up.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_note",
            "description": "Save an idea, finding, or insight to the knowledge graph for future reference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The note content to save.",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Optional topic tag for the note.",
                    },
                },
                "required": ["content"],
            },
        },
    },
]


async def dispatch_tool(
    name: str,
    arguments: str,
    *,
    config: WeaverConfig,
    graph: GraphClient,
    group_id: str | None = None,
) -> str:
    """Execute a tool call and return the result as a JSON string."""
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return json.dumps({"error": f"Invalid JSON arguments: {arguments}"})

    try:
        if name == "web_search":
            result = await web_search(args["query"], searxng_url=config.searxng_url)
        elif name == "read_url":
            result = await read_url(args["url"])
        elif name == "graph_query":
            result = await graph_query(graph, args["query"], group_id=group_id, limit=config.graph_search_results)
        elif name == "graph_note":
            result = await graph_note(graph, args["content"], topic=args.get("topic"), group_id=group_id)
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        log.warning("tool %s failed: %s", name, e)
        result = {"error": str(e)}

    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)
