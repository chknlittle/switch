"""Miscellaneous session commands."""

from __future__ import annotations

from src.commands.mixins.base import CommandMixinBase
from src.commands.registry import command


class MiscCommandsMixin(CommandMixinBase):
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
