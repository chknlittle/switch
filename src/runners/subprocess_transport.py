"""Subprocess transport helpers for runners."""

from __future__ import annotations

import asyncio


class SubprocessTransport:
    def __init__(self):
        self.process: asyncio.subprocess.Process | None = None

    async def start(
        self,
        cmd: list[str],
        *,
        cwd: str,
        stdout_limit: int,
    ) -> asyncio.StreamReader:
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            limit=stdout_limit,
        )

        if self.process.stdout is None:
            raise RuntimeError("Subprocess stdout missing")

        return self.process.stdout

    async def wait(self) -> int:
        if not self.process:
            return 0
        await self.process.wait()
        return int(self.process.returncode or 0)

    def cancel(self) -> None:
        if self.process:
            self.process.terminate()
