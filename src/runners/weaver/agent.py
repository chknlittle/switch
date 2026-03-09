"""Weaver agent loop: prompt -> LLM -> tool calls -> loop."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from src.runners.ports import RunnerEvent
from src.runners.weaver.config import WeaverConfig
from src.runners.weaver.graph import GraphClient
from src.runners.weaver.llm import chat_completion
from src.runners.weaver.tools.registry import TOOL_DEFINITIONS, dispatch_tool

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are Weaver, a research and brainstorming agent. Your purpose is to help \
explore ideas, research topics, and build a shared knowledge graph that \
persists across conversations.

You have access to these tools:
- **web_search**: Search the web for current information
- **read_url**: Read the full content of a web page
- **graph_query**: Search the knowledge graph for previously stored facts and ideas
- **graph_note**: Save important findings, ideas, or insights to the knowledge graph

When researching a topic:
1. First check the knowledge graph for existing context
2. Search the web for new information
3. Read relevant pages for details
4. Save key findings to the graph for future reference

Be thorough but concise. Think step by step. When brainstorming, explore \
multiple angles and connections. Always cite your sources when using web results.\
"""


class WeaverAgent:
    """Runs the agent tool-calling loop."""

    def __init__(
        self,
        config: WeaverConfig,
        client: AsyncOpenAI,
        graph: GraphClient,
    ):
        self.config = config
        self.client = client
        self.graph = graph
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    async def run(
        self,
        prompt: str,
        *,
        group_id: str | None = None,
    ) -> AsyncIterator[RunnerEvent]:
        start = time.monotonic()
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Build initial messages
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Retrieve graph context
        graph_context = await self.graph.search(
            prompt, group_id=group_id, limit=self.config.graph_search_results
        )
        if graph_context:
            ctx_text = "Relevant knowledge from previous sessions:\n"
            for item in graph_context:
                if item.get("type") == "entity":
                    ctx_text += f"- Entity: {item.get('name', '')}: {item.get('summary', '') or item.get('fact', '')}\n"
                else:
                    ctx_text += f"- Relation: {item.get('name', '')}: {item.get('fact', '')}\n"
            messages.append({"role": "system", "content": ctx_text})

        messages.append({"role": "user", "content": prompt})

        # Agent loop
        for round_num in range(self.config.max_tool_rounds):
            if self._cancelled:
                yield ("cancelled", None)
                return

            try:
                response = await chat_completion(
                    self.client,
                    self.config.llm_model,
                    messages,
                    tools=TOOL_DEFINITIONS,
                )
            except Exception as e:
                log.error("LLM call failed: %s", e)
                yield ("error", f"LLM error: {e}")
                return

            usage = response.get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

            tool_calls = response.get("tool_calls", [])

            # If no tool calls, this is the final text response
            if not tool_calls:
                content = response.get("content") or ""
                yield ("text", content)
                break

            # Append assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": response.get("content"),
                "tool_calls": tool_calls,
            })

            # Execute each tool call
            for tc in tool_calls:
                if self._cancelled:
                    yield ("cancelled", None)
                    return

                func = tc.get("function", {})
                tool_name = func.get("name", "unknown")
                tool_args = func.get("arguments", "{}")
                tool_id = tc.get("id", "")

                yield ("tool", f"[tool:{tool_name} {_summarize_args(tool_args)}]")

                result = await dispatch_tool(
                    tool_name,
                    tool_args,
                    config=self.config,
                    graph=self.graph,
                    group_id=group_id,
                )

                yield ("tool_result", _truncate(result, 500))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result,
                })

        duration = time.monotonic() - start

        # Background: index the exchange into the graph
        exchange_text = f"User: {prompt}\nAssistant: {response.get('content', '')}"
        asyncio.create_task(_background_index(self.graph, exchange_text, group_id))

        yield ("result", {
            "tokens_in": total_prompt_tokens,
            "tokens_out": total_completion_tokens,
            "tokens_total": total_prompt_tokens + total_completion_tokens,
            "cost_usd": 0,  # local model
            "duration_s": round(duration, 2),
        })


async def _background_index(graph: GraphClient, content: str, group_id: str | None) -> None:
    """Fire-and-forget graph indexing."""
    try:
        await graph.add_episode(content, group_id=group_id, source="exchange")
    except Exception:
        log.debug("background graph indexing failed", exc_info=True)


def _summarize_args(args_json: str) -> str:
    try:
        args = json.loads(args_json)
        parts = []
        for v in args.values():
            s = str(v)
            if len(s) > 60:
                s = s[:57] + "..."
            parts.append(s)
        return " ".join(parts)
    except Exception:
        return args_json[:60]


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."
