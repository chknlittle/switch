"""Minimal JSON-RPC client for Cursor Agent ACP over stdio."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

log = logging.getLogger("cursor.acp")


class CursorACPClient:
    def __init__(self, argv: list[str], *, cwd: str):
        self.argv = argv
        self.cwd = cwd
        self.proc: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._stdout_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None

    async def start(self) -> None:
        # Cursor ACP can emit very large single-line JSON-RPC messages
        # (e.g. available commands, tool payloads). asyncio's default
        # StreamReader limit is 64 KiB, which raises LimitOverrunError before
        # readline() sees the newline. Keep this comfortably above expected ACP
        # event sizes while still bounded.
        self.proc = await asyncio.create_subprocess_exec(
            *self.argv,
            cwd=self.cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=16 * 1024 * 1024,
        )
        self._stdout_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())

    async def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                return
            try:
                msg = json.loads(line.decode(errors="replace"))
            except Exception:
                log.debug("Ignoring non-JSON ACP stdout: %r", line[:500])
                continue
            if not isinstance(msg, dict):
                continue
            msg_id = msg.get("id")
            if msg_id is not None and ("result" in msg or "error" in msg):
                fut = self._pending.pop(int(msg_id), None)
                if fut and not fut.done():
                    if "error" in msg:
                        fut.set_exception(RuntimeError(json.dumps(msg["error"])))
                    else:
                        fut.set_result(msg.get("result"))
                continue
            await self.events.put(msg)

    async def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                return
            log.info("Cursor ACP stderr: %s", line.decode(errors="replace").rstrip())

    async def request(self, method: str, params: dict[str, Any], *, timeout_s: float) -> Any:
        assert self.proc and self.proc.stdin
        msg_id = self._next_id
        self._next_id += 1
        fut = asyncio.get_running_loop().create_future()
        self._pending[msg_id] = fut
        payload = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        self.proc.stdin.write((json.dumps(payload) + "\n").encode())
        await self.proc.stdin.drain()
        return await asyncio.wait_for(fut, timeout=timeout_s)

    async def respond(self, msg_id: int, result: dict[str, Any]) -> None:
        assert self.proc and self.proc.stdin
        payload = {"jsonrpc": "2.0", "id": msg_id, "result": result}
        self.proc.stdin.write((json.dumps(payload) + "\n").encode())
        await self.proc.stdin.drain()

    def terminate(self) -> None:
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()

    async def close(self) -> None:
        if not self.proc:
            return
        if self.proc.stdin:
            self.proc.stdin.close()
        if self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()
        for task in (self._stdout_task, self._stderr_task):
            if task and not task.done():
                task.cancel()
