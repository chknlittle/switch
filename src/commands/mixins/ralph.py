"""Ralph loop commands."""

from __future__ import annotations

from src.commands.mixins.base import CommandMixinBase
from src.commands.registry import command
from src.core.session_runtime.api import RalphConfig
from src.lifecycle.sessions import create_session as lifecycle_create_session
from src.ralph import parse_ralph_command


class RalphCommandsMixin(CommandMixinBase):
    @command("/ralph-cancel", "/ralph-stop")
    async def ralph_cancel(self, _body: str) -> bool:
        """Cancel Ralph loop."""
        if self.bot.session.request_ralph_stop():
            self.bot.send_reply("Ralph loop will stop after current iteration...")
            return True

        self.bot.send_reply("No Ralph loop running.")
        return True

    @command("/ralph-status")
    async def ralph_status(self, _body: str) -> bool:
        """Show Ralph loop status."""
        live = self.bot.session.get_ralph_status()
        if live and live.status in {"queued", "running", "stopping"}:
            max_str = (
                str(live.max_iterations) if live.max_iterations > 0 else "unlimited"
            )
            wait_minutes = float(live.wait_seconds or 0.0) / 60.0
            self.bot.send_reply(
                f"Ralph {live.status.upper()}\n"
                f"Iteration: {live.current_iteration}/{max_str}\n"
                f"Cost so far: ${live.total_cost:.3f}\n"
                f"Wait: {wait_minutes:.2f} min\n"
                f"Promise: {live.completion_promise or 'none'}"
            )
            return True

        loop = self.bot.ralph_loops.get_latest(self.bot.session_name)
        if loop:
            max_str = str(loop.max_iterations) if loop.max_iterations else "unlimited"
            wait_minutes = loop.wait_seconds / 60.0
            self.bot.send_reply(
                f"Last Ralph: {loop.status}\n"
                f"Iterations: {loop.current_iteration}/{max_str}\n"
                f"Wait: {wait_minutes:.2f} min\n"
                f"Cost: ${loop.total_cost:.3f}"
            )
            return True

        self.bot.send_reply("No Ralph loops in this session.")
        return True

    @command("/ralph", exact=False)
    async def ralph(self, body: str) -> bool:
        """Start a Ralph loop."""
        ralph_args = parse_ralph_command(body)
        if ralph_args is None:
            self.bot.send_reply(
                "Usage: /ralph <prompt> [--max N] [--done 'promise'] [--wait MINUTES]\n"
                "                 [--look]  (prompt-only: no cross-iteration context)\n"
                "                 [--swarm N]  (start N parallel Ralph sessions)\n"
                "  or:  /ralph <N> <prompt>  (shorthand)\n\n"
                "Examples:\n"
                "  /ralph 20 Fix all type errors\n"
                "  /ralph Refactor auth --max 10 --wait 5 --done 'All tests pass'\n"
                "  /ralph Refactor auth --max 10 --swarm 5\n\n"
                "Notes:\n"
                "  --wait is in minutes (e.g. 0.5 = 30 seconds).\n"
                "Commands:\n"
                "  /ralph-status - check progress\n"
                "  /ralph-cancel - stop loop"
            )
            return True

        swarm = int(ralph_args.get("swarm") or 1)
        if swarm > 1:
            if not self.bot.manager:
                self.bot.send_reply("Swarm requires a session manager (try from the dispatcher contact).")
                return True

            MAX_SWARM = 50
            if swarm > MAX_SWARM:
                swarm = MAX_SWARM
                self.bot.send_reply(f"Clamped --swarm to {MAX_SWARM} for safety.")

            forward_args = (ralph_args.get("forward_args") or "").strip()
            if not forward_args:
                self.bot.send_reply("Invalid /ralph args (empty after --swarm).")
                return True

            parent = self.bot.sessions.get(self.bot.session_name)
            engine = parent.active_engine if parent else "pi"
            model_id = parent.model_id if parent else None

            names: list[str] = []
            for _ in range(swarm):
                created_name = await lifecycle_create_session(
                    self.bot.manager,
                    "",
                    engine=engine,
                    model_id=model_id,
                    label=None,
                    name_hint="ralph",
                    announce="Ralph session '{name}'. Starting loop...",
                    dispatcher_jid=None,
                )
                if not created_name:
                    continue
                bot = self.bot.manager.session_bots.get(created_name)
                if not bot:
                    continue
                await bot.commands.handle(f"/ralph {forward_args}")
                names.append(created_name)

            if not names:
                self.bot.send_reply("Failed to create Ralph swarm sessions.")
                return True

            self.bot.send_reply(
                "\n".join(
                    [
                        f"Started Ralph swarm x{len(names)}:",
                        *[f"  {n}@{self.bot.xmpp_domain}" for n in names],
                    ]
                )
            )
            return True

        if self.bot.processing or self.bot.session.pending_count() > 0:
            self.bot.send_reply(
                "Already running or queued. Use /ralph-cancel (or /cancel) first."
            )
            return True

        await self.bot.session.start_ralph(
            RalphConfig(
                prompt=ralph_args["prompt"],
                max_iterations=int(ralph_args["max_iterations"] or 0),
                completion_promise=ralph_args["completion_promise"],
                wait_seconds=float(ralph_args["wait_minutes"] or 0.0) * 60.0,
                prompt_only=bool(ralph_args.get("prompt_only")),
            )
        )
        return True

    @command("/ralph-look", "/ralphlook", exact=False)
    async def ralph_look(self, body: str) -> bool:
        """Start a prompt-only Ralph loop (fresh context every iteration)."""
        raw = body.strip()
        low = raw.lower()
        if low.startswith("/ralph-look"):
            rest = raw[len("/ralph-look") :].strip()
        else:
            rest = raw[len("/ralphlook") :].strip()

        if not rest:
            self.bot.send_reply(
                "Usage: /ralph-look <prompt> [--max N] [--done 'promise'] [--wait MINUTES]\n"
                "  or:  /ralph-look <N> <prompt>  (shorthand)"
            )
            return True

        # Delegate to /ralph with --look forced on.
        return await self.ralph(f"/ralph {rest} --look")
