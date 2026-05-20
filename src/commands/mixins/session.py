"""Session lifecycle and introspection commands."""

from __future__ import annotations

from typing import Any, cast

from src.commands.mixins.base import CommandMixinBase
from src.commands.registry import command
from src.engines import get_engine_spec
from src.runners.pi.runner import PiRunner


class SessionCommandsMixin(CommandMixinBase):
    @command("/dispatchers", "/delegates")
    async def dispatchers(self, _body: str) -> bool:
        """List configured dispatchers available for delegation."""
        manager = self.bot.manager
        if not manager:
            self.bot.send_reply("No session manager attached; cannot list dispatchers.")
            return True

        cfg = manager.dispatchers_config or {}
        if not cfg:
            self.bot.send_reply("No dispatchers configured.")
            return True

        lines: list[str] = ["Available dispatchers:"]
        for name in sorted(cfg.keys()):
            raw = cfg.get(name) or {}
            if not isinstance(raw, dict):
                continue
            if raw.get("disabled") is True:
                continue
            jid = str(raw.get("jid") or "").strip()
            has_password = bool(str(raw.get("password") or "").strip())
            if not jid:
                continue
            suffix = "" if has_password else " (missing password)"
            lines.append(f"- {name} ({jid}){suffix}")

        if len(lines) == 1:
            self.bot.send_reply("No active dispatchers configured.")
            return True

        lines.append("Example: ask oc-gemini What do you think about this plan?")
        self.bot.send_reply("\n".join(lines))
        return True

    @command("/kill")
    async def kill(self, _body: str) -> bool:
        """Hard-kill the session (cancel work, close account, stop reconnect)."""
        # Send ack before we start teardown (account deletion can race delivery).
        self.bot.send_reply("Killing session (hard kill)...")
        self.bot.spawn_guarded(self.bot.hard_kill(), context="session.hard_kill")
        return True

    @command("/cancel")
    async def cancel(self, _body: str) -> bool:
        """Cancel current operation."""
        cancelled = self.bot.cancel_operations(notify=False, hard_abort_vllm=True)
        if cancelled:
            self.bot.send_reply("Cancelling current work...")
        else:
            self.bot.send_reply("Nothing running to cancel.")
        return True

    @command("/peek", exact=False)
    async def peek(self, body: str) -> bool:
        """Show recent output."""
        parts = body.strip().lower().split()
        num_lines = 30
        if len(parts) > 1:
            try:
                num_lines = int(parts[1])
            except ValueError:
                pass
        await self.bot.peek_output(num_lines)
        return True

    @command("/reset")
    async def reset(self, _body: str) -> bool:
        """Reset session context."""
        session = self.bot.sessions.get(self.bot.session_name)
        if not session:
            self.bot.send_reply("Session not found.")
            return True

        # Cancel any in-flight work before clearing remote session state.
        self.bot.cancel_operations(notify=False)

        engine = (session.active_engine or "").strip().lower()
        spec = get_engine_spec(engine)
        if not spec:
            self.bot.send_reply(f"Unknown engine '{session.active_engine}'.")
            return True
        await self.bot.sessions.reset_remote_session(self.bot.session_name, engine)
        self.bot.send_reply("Session reset.")
        return True

    @command("/compact")
    async def compact(self, _body: str) -> bool:
        """Compact Pi's context window."""
        runner = cast(Any, self.bot.session).runner
        if not isinstance(runner, PiRunner):
            self.bot.send_reply("Not a Pi session — /compact only works with Pi.")
            return True
        if not self.bot.processing:
            self.bot.send_reply("No active Pi process to compact.")
            return True
        sent = await runner.compact()
        if sent:
            self.bot.send_reply("Compacting context...")
        else:
            self.bot.send_reply("Failed to send compact (process not running).")
        return True
