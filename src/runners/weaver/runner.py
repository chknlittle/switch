"""WeaverRunner — implements the Runner protocol with an internal agent loop."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from src.runners.ports import RunnerEvent
from src.runners.weaver.agent import WeaverAgent
from src.runners.weaver.config import WeaverConfig
from src.runners.weaver.graph import GraphClient
from src.runners.weaver.llm import create_client

log = logging.getLogger(__name__)


class WeaverRunner:
    """Runner that uses an internal agent loop with tool calling."""

    def __init__(
        self,
        working_dir: str,
        output_dir: Path,
        session_name: str | None = None,
        *,
        config: WeaverConfig | None = None,
    ):
        self.working_dir = working_dir
        self.output_dir = output_dir
        self.session_name = session_name
        self.config = config or WeaverConfig()
        self._client = create_client(self.config)
        self._graph = GraphClient(self.config)
        self._agent: WeaverAgent | None = None
        self._graph_initialized = False
        self._graph_init_lock = asyncio.Lock()

    async def run(
        self, prompt: str, session_id: str | None = None
    ) -> AsyncIterator[RunnerEvent]:
        async with self._graph_init_lock:
            if not self._graph_initialized:
                try:
                    await self._graph.init()
                    self._graph_initialized = True
                except Exception:
                    log.warning("graph init failed, will retry next call", exc_info=True)

        self._agent = WeaverAgent(self.config, self._client, self._graph)

        # Use session_id as group_id for graph namespacing
        group_id = session_id or self.session_name

        async for event in self._agent.run(prompt, group_id=group_id):
            yield event

        # Yield session_id so the runtime can persist it
        if group_id:
            yield ("session_id", group_id)

    def cancel(self) -> None:
        if self._agent:
            self._agent.cancel()

    async def cleanup(self) -> None:
        await self._graph.close()
