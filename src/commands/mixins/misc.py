"""Miscellaneous session commands."""

from __future__ import annotations

import asyncio
import os

from src.commands.mixins.base import CommandMixinBase
from src.commands.registry import command

_ACORN_RESET_TIMEOUT_S = 60


class MiscCommandsMixin(CommandMixinBase):
    @command("/acorn-reset", "/reset-acorn")
    async def acorn_reset(self, _body: str) -> bool:
        """Reset Acorn/Hermes conversation state on helga (no context read)."""
        host = os.getenv("ACORN_RESET_HOST", "192.168.0.129")
        user = os.getenv("ACORN_RESET_USER", "rin")
        stack = os.getenv("ACORN_RESET_STACK", "~/hermes-stack")
        cmd = (
            f'ssh -o ConnectTimeout=10 -o BatchMode=yes {user}@{host} '
            f'"cd {stack} && ./scripts/reset-state.sh"'
        )

        self.bot.send_typing()
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=_ACORN_RESET_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            self.bot.send_reply(
                f"Acorn reset timed out after {_ACORN_RESET_TIMEOUT_S}s."
            )
            return True
        except OSError as err:
            self.bot.send_reply(f"Acorn reset failed to run: {err}")
            return True

        output = stdout.decode("utf-8", errors="replace").strip()
        if proc.returncode == 0:
            self.bot.send_reply("Acorn context refreshed. Hermes + Acorn restarted.")
        else:
            detail = output[:1000] if output else "(no output)"
            self.bot.send_reply(
                f"Acorn reset failed (exit {proc.returncode}):\n{detail}"
            )
        return True

    @command("/call")
    async def call(self, _body: str) -> bool:
        """Show active voice call status."""
        voice = getattr(self.bot, "_voice", None)
        if voice is None:
            self.bot.send_reply("Voice calls are not enabled (set SWITCH_VOICE_ENABLED=1).")
            return True

        count = voice.active_call_count
        if count == 0:
            self.bot.send_reply("No active voice calls.")
        else:
            sids = voice.active_call_sids
            lines = [f"Active voice calls: {count}"]
            for sid in sids:
                lines.append(f"  - {sid}")
            self.bot.send_reply("\n".join(lines))
        return True
