"""Cursor Agent ACP runner."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from src.runners.base import BaseRunner, RunState
from src.runners.cursor.acp import CursorACPClient
from src.runners.cursor.config import CursorConfig
from src.runners.ports import RunnerEvent

log = logging.getLogger("cursor.runner")

Event = RunnerEvent


class CursorACPRunner(BaseRunner):
    """Runs Cursor Agent through ACP (`agent acp`) over stdio."""

    def __init__(
        self,
        working_dir: str,
        output_dir: Path,
        session_name: str | None = None,
        config: CursorConfig | None = None,
    ):
        super().__init__(working_dir, output_dir, session_name)
        self._config = config or CursorConfig()
        self._client: CursorACPClient | None = None
        self._cancelled = False

    def _argv(self) -> list[str]:
        return [self._config.resolve_bin(), "--model", self._config.resolve_model(), "acp"]

    async def _ensure_started(self) -> CursorACPClient:
        client = CursorACPClient(self._argv(), cwd=self.working_dir)
        await client.start()
        timeout = self._config.request_timeout_s
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
        self._client = client
        return client

    async def _open_session(self, client: CursorACPClient, session_id: str | None) -> str:
        timeout = self._config.request_timeout_s
        params: dict[str, Any] = {"cwd": self.working_dir, "mcpServers": []}
        if session_id:
            # Cursor docs advertise session/load but are sparse on payload details.
            # If the local CLI rejects this shape, fall back to a fresh session.
            try:
                result = await client.request(
                    "session/load",
                    {"sessionId": session_id, **params},
                    timeout_s=timeout,
                )
                if isinstance(result, dict):
                    return session_id
            except Exception:
                log.warning("Cursor session/load failed; starting a new session", exc_info=True)
        result = await client.request("session/new", params, timeout_s=timeout)
        if not isinstance(result, dict) or not result.get("sessionId"):
            raise RuntimeError(f"Cursor session/new did not return sessionId: {result!r}")
        return str(result["sessionId"])

    async def _permission_auto_allow(self, client: CursorACPClient, msg: dict[str, Any]) -> None:
        msg_id = msg.get("id")
        if msg_id is None:
            return
        await client.respond(
            int(msg_id),
            {"outcome": {"outcome": "selected", "optionId": self._config.permission_option_id}},
        )

    def _tool_summary(self, update: dict[str, Any]) -> str | None:
        kind = update.get("sessionUpdate")
        if kind in {"tool_call", "tool_call_update", "tool_call_started"}:
            name = update.get("name") or update.get("toolName") or update.get("title") or "tool"
            return f"Cursor: {name}"
        if kind in {"command_started", "command_update"}:
            cmd = update.get("command") or update.get("text") or "command"
            return f"Cursor shell: {cmd}"
        return None

    async def run(self, prompt: str, session_id: str | None = None) -> AsyncIterator[Event]:
        state = RunState(start_time=datetime.now())
        self._cancelled = False
        self._log_prompt(prompt)
        client: CursorACPClient | None = None
        prompt_task: asyncio.Task | None = None
        try:
            client = await self._ensure_started()
            cursor_session_id = await self._open_session(client, session_id)
            state.session_id = cursor_session_id
            yield ("session_id", cursor_session_id)

            prompt_task = asyncio.create_task(
                client.request(
                    "session/prompt",
                    {"sessionId": cursor_session_id, "prompt": [{"type": "text", "text": prompt}]},
                    timeout_s=self._config.request_timeout_s,
                )
            )

            while True:
                if self._cancelled:
                    yield ("cancelled", None)
                    return
                if prompt_task.done() and client.events.empty():
                    result = prompt_task.result()
                    state.saw_result = True
                    if state.text:
                        self._log_response(state.text)
                    yield (
                        "result",
                        {
                            "duration_s": state.duration_s,
                            "session_id": cursor_session_id,
                            "stop_reason": result.get("stopReason") if isinstance(result, dict) else None,
                        },
                    )
                    return
                try:
                    msg = await asyncio.wait_for(client.events.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                method = msg.get("method")
                if method == "session/request_permission":
                    await self._permission_auto_allow(client, msg)
                    continue
                if method != "session/update":
                    continue
                params = msg.get("params") or {}
                if params.get("sessionId") not in {None, cursor_session_id}:
                    continue
                update = params.get("update") or {}
                if update.get("sessionUpdate") == "agent_message_chunk":
                    text = ((update.get("content") or {}).get("text") or "")
                    if text:
                        state.text += text
                        yield ("text", text)
                elif update.get("sessionUpdate") == "agent_thought_chunk":
                    # Thought chunks are not user-visible in Switch chat.
                    continue
                else:
                    summary = self._tool_summary(update)
                    if summary:
                        yield ("tool", summary)
        except Exception as e:
            log.exception("Cursor ACP runner error")
            yield ("error", str(e))
        finally:
            if prompt_task and not prompt_task.done():
                prompt_task.cancel()

    def cancel(self) -> None:
        self._cancelled = True
        if self._client:
            self._client.terminate()

    async def cleanup(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
