"""Cursor ACP transport: client lifecycle, session open, prompt task."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.runners.cursor.acp import CursorACPClient
from src.runners.cursor.config import CursorConfig
from src.runners.timeouts import control_plane_timeout_s

log = logging.getLogger("cursor.transport")


class CursorACPTransport:
    def __init__(self, config: CursorConfig):
        self._config = config
        self._client: CursorACPClient | None = None
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def client(self) -> CursorACPClient | None:
        return self._client

    async def start(self, *, argv: list[str], cwd: str) -> CursorACPClient:
        client = CursorACPClient(argv, cwd=cwd)
        await client.start()
        self._client = client
        await self._initialize(client)
        return client

    async def _initialize(self, client: CursorACPClient) -> None:
        timeout = control_plane_timeout_s(
            override=self._config.control_plane_timeout_s,
        )
        await client.request(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                },
                "clientInfo": {"name": "switch-cursor-acp", "version": "0.1.0"},
            },
            timeout_s=timeout,
        )
        await client.request(
            "authenticate",
            {"methodId": self._config.auth_method},
            timeout_s=timeout,
        )

    async def open_session(
        self,
        client: CursorACPClient,
        *,
        session_id: str | None,
        cwd: str,
    ) -> str:
        timeout = control_plane_timeout_s(
            override=self._config.control_plane_timeout_s,
        )
        params: dict[str, Any] = {"cwd": cwd, "mcpServers": []}
        if session_id:
            try:
                result = await client.request(
                    "session/load",
                    {"sessionId": session_id, **params},
                    timeout_s=timeout,
                )
                if isinstance(result, dict):
                    return session_id
            except Exception:
                log.warning(
                    "Cursor session/load failed; starting a new session",
                    exc_info=True,
                )

        result = await client.request("session/new", params, timeout_s=timeout)
        if not isinstance(result, dict) or not result.get("sessionId"):
            raise RuntimeError(f"Cursor session/new did not return sessionId: {result!r}")
        return str(result["sessionId"])

    def start_prompt(
        self,
        client: CursorACPClient,
        *,
        session_id: str,
        prompt: str,
    ) -> asyncio.Task[Any]:
        # Agent turns stream on a separate notification channel. The prompt RPC
        # must not use a wall-clock cap — completion is driven by the shared
        # queue pipeline (trailing-event idle timeout + finalize).
        return asyncio.create_task(
            client.request(
                "session/prompt",
                {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": prompt}],
                },
                timeout_s=None,
            )
        )

    async def allow_permission(self, msg: dict[str, Any]) -> None:
        client = self._client
        if not client:
            return
        msg_id = msg.get("id")
        if msg_id is None:
            return
        await client.respond(
            int(msg_id),
            {
                "outcome": {
                    "outcome": "selected",
                    "optionId": self._config.permission_option_id,
                }
            },
        )

    def reader_task(self) -> asyncio.Task | None:
        client = self._client
        if client is None:
            return None
        return client.reader_task

    def cancel(self) -> None:
        self._cancelled = True
        if self._client:
            self._client.terminate()

    async def cleanup(self, *, prompt_task: asyncio.Task | None) -> None:
        if prompt_task and not prompt_task.done():
            prompt_task.cancel()
        if self._client:
            await self._client.close()
            self._client = None
