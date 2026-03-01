"""Command handlers for session bot."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast

from src.engines import normalize_engine
from src.core.session_runtime.api import RalphConfig
from src.lifecycle.sessions import create_session as lifecycle_create_session
from src.ralph import parse_ralph_command
from src.runners.pi.runner import PiRunner

if TYPE_CHECKING:
    from src.bots.session import SessionBot


def command(name: str, *aliases: str, exact: bool = True):
    """Decorator to register a command handler.

    Args:
        name: Primary command name (e.g., "/kill")
        *aliases: Additional names that trigger this command
        exact: If True, requires exact match; if False, allows prefix match
    """

    def decorator(
        func: Callable[..., Awaitable[bool]],
    ) -> Callable[..., Awaitable[bool]]:
        setattr(func, "_command_name", name)
        setattr(func, "_command_aliases", aliases)
        setattr(func, "_command_exact", exact)
        return func

    return decorator


class CommandHandler:
    """Handles slash commands for a session bot.

    Commands are registered via the @command decorator on methods.
    The handler auto-discovers all decorated methods on init.
    """

    def __init__(self, bot: "SessionBot"):
        self.bot = bot
        self._commands: dict[str, tuple[Callable[..., Awaitable[bool]], bool]] = {}
        self._discover_commands()

    def _discover_commands(self) -> None:
        """Find all @command decorated methods and register them."""
        for name in dir(self):
            method = getattr(self, name)
            if callable(method) and hasattr(method, "_command_name"):
                m = cast(Any, method)
                cmd_name = cast(str, m._command_name)
                exact = cast(bool, m._command_exact)
                handler = cast(Callable[..., Awaitable[bool]], method)
                self._commands[cmd_name] = (handler, exact)
                for alias in cast(tuple[str, ...], m._command_aliases):
                    self._commands[alias] = (handler, exact)

    async def handle(self, body: str) -> bool:
        """Handle a command. Returns True if command was handled."""
        cmd = body.strip().lower()

        # Exact matches first.
        for prefix, (handler, exact) in self._commands.items():
            if exact and cmd == prefix:
                return await handler(body)

        # Then prefix matches, preferring the longest prefix (avoids overlaps like
        # /ralph vs /ralph-look).
        best: tuple[int, Callable[..., Awaitable[bool]]] | None = None
        for prefix, (handler, exact) in self._commands.items():
            if exact:
                continue
            if cmd.startswith(prefix):
                score = len(prefix)
                if best is None or score > best[0]:
                    best = (score, handler)
        if best is not None:
            return await best[1](body)

        return False

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
        cancelled = self.bot.cancel_operations(notify=False)
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

    @command("/agent", exact=False)
    async def agent(self, body: str) -> bool:
        """Switch active engine."""
        parts = body.strip().lower().split()
        if len(parts) < 2:
            self.bot.send_reply("Usage: /agent oc|cc|pi")
            return True

        engine = normalize_engine(parts[1])
        if not engine:
            self.bot.send_reply("Usage: /agent oc|cc|pi")
            return True

        self.bot.sessions.update_engine(self.bot.session_name, engine)
        self.bot.send_reply(f"Active engine set to {engine}.")
        return True

    @command("/thinking", exact=False)
    async def thinking(self, body: str) -> bool:
        """Set reasoning mode."""
        parts = body.strip().lower().split()
        if len(parts) < 2 or parts[1] not in ("normal", "high"):
            self.bot.send_reply("Usage: /thinking normal|high")
            return True

        session = self.bot.sessions.get(self.bot.session_name)
        if not session:
            self.bot.send_reply("Session not found.")
            return True

        engine = (session.active_engine or "").strip().lower()
        if engine not in {"opencode", "pi"}:
            self.bot.send_reply("/thinking only applies to OpenCode/Pi sessions.")
            return True

        self.bot.sessions.update_reasoning_mode(self.bot.session_name, parts[1])
        self.bot.send_reply(f"Reasoning mode set to {parts[1]}.")
        return True

    @command("/model", exact=False)
    async def model(self, body: str) -> bool:
        """Set model ID."""
        parts = body.strip().split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            self.bot.send_reply("Usage: /model <model-id>")
            return True

        model_id = parts[1].strip()
        self.bot.sessions.update_model(self.bot.session_name, model_id)
        self.bot.send_reply(f"Model set to {model_id}.")
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
        if engine == "claude":
            self.bot.sessions.reset_claude_session(self.bot.session_name)
        elif engine == "opencode":
            self.bot.sessions.reset_opencode_session(self.bot.session_name)
        elif engine == "pi":
            self.bot.sessions.reset_pi_session(self.bot.session_name)
        else:
            self.bot.send_reply(f"Unknown engine '{session.active_engine}'.")
            return True
        self.bot.send_reply("Session reset.")
        return True

    @command("/compact")
    async def compact(self, _body: str) -> bool:
        """Compact Pi's context window."""
        runner = self.bot.session.runner
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
            agent = parent.opencode_agent if parent else "bridge"
            model_id = parent.model_id if parent else None

            names: list[str] = []
            for _ in range(swarm):
                created_name = await lifecycle_create_session(
                    self.bot.manager,
                    "",
                    engine=engine,
                    opencode_agent=agent,
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
