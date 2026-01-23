"""Command handlers for session bot."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast

from src.engines import normalize_engine
from src.ralph import RalphLoop, parse_ralph_command

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
        func._command_name = name
        func._command_aliases = aliases
        func._command_exact = exact
        return func

    return decorator


class CommandHandler:
    """Handles slash commands for a session bot.

    Commands are registered via the @command decorator on methods.
    The handler auto-discovers all decorated methods on init.
    """

    def __init__(self, bot: "SessionBot"):
        self.bot = bot
        self._commands: dict[str, tuple[Callable[[str], Awaitable[bool]], bool]] = {}
        self._discover_commands()

    def _discover_commands(self) -> None:
        """Find all @command decorated methods and register them."""
        for name in dir(self):
            method = getattr(self, name)
            if callable(method) and hasattr(method, "_command_name"):
                cmd_name = method._command_name
                exact = method._command_exact
                self._commands[cmd_name] = (method, exact)
                for alias in method._command_aliases:
                    self._commands[alias] = (method, exact)

    async def handle(self, body: str) -> bool:
        """Handle a command. Returns True if command was handled."""
        cmd = body.strip().lower()

        # Try exact matches first
        for prefix, (handler, exact) in self._commands.items():
            if exact and cmd == prefix:
                return await handler(body)
            if not exact and cmd.startswith(prefix):
                return await handler(body)

        return False

    @command("/kill")
    async def kill(self, _body: str) -> bool:
        """End the session."""
        self.bot.send_reply("Ending session. Goodbye!")
        asyncio.ensure_future(self.bot._self_destruct())
        return True

    @command("/cancel")
    async def cancel(self, _body: str) -> bool:
        """Cancel current operation."""
        if self.bot.ralph_loop:
            self.bot.ralph_loop.cancel()
            if self.bot.runner:
                self.bot.runner.cancel()
            self.bot.send_reply("Cancelling Ralph loop...")
        elif self.bot.runner and self.bot.processing:
            self.bot.runner.cancel()
            self.bot.send_reply("Cancelling current run...")
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
            self.bot.send_reply("Usage: /agent oc|cc")
            return True

        engine = normalize_engine(parts[1])
        if not engine:
            self.bot.send_reply("Usage: /agent oc|cc")
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

        handler = self.bot.engine_handler_for(session.active_engine)
        if not handler:
            self.bot.send_reply(f"Unknown engine '{session.active_engine}'.")
            return True
        if not handler.supports_reasoning:
            self.bot.send_reply("/thinking only applies to OpenCode sessions.")
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

        handler = self.bot.engine_handler_for(session.active_engine)
        if not handler:
            self.bot.send_reply(f"Unknown engine '{session.active_engine}'.")
            return True
        handler.reset()
        self.bot.send_reply("Session reset.")
        return True

    @command("/ralph-cancel", "/ralph-stop")
    async def ralph_cancel(self, _body: str) -> bool:
        """Cancel Ralph loop."""
        if self.bot.ralph_loop:
            self.bot.ralph_loop.cancel()
            self.bot.send_reply("Ralph loop will stop after current iteration...")
        else:
            self.bot.send_reply("No Ralph loop running.")
        return True

    @command("/ralph-status")
    async def ralph_status(self, _body: str) -> bool:
        """Show Ralph loop status."""
        if self.bot.ralph_loop:
            rl = self.bot.ralph_loop
            max_str = str(rl.max_iterations) if rl.max_iterations > 0 else "unlimited"
            self.bot.send_reply(
                f"Ralph RUNNING\n"
                f"Iteration: {rl.current_iteration}/{max_str}\n"
                f"Cost so far: ${rl.total_cost:.3f}\n"
                f"Promise: {rl.completion_promise or 'none'}"
            )
        else:
            loop = self.bot.ralph_loops.get_latest(self.bot.session_name)
            if loop:
                max_str = (
                    str(loop.max_iterations) if loop.max_iterations else "unlimited"
                )
                self.bot.send_reply(
                    f"Last Ralph: {loop.status}\n"
                    f"Iterations: {loop.current_iteration}/{max_str}\n"
                    f"Cost: ${loop.total_cost:.3f}"
                )
            else:
                self.bot.send_reply("No Ralph loops in this session.")
        return True

    @command("/ralph", exact=False)
    async def ralph(self, body: str) -> bool:
        """Start a Ralph loop."""
        ralph_args = parse_ralph_command(body)
        if ralph_args is None:
            self.bot.send_reply(
                "Usage: /ralph <prompt> [--max N] [--done 'promise']\n"
                "  or:  /ralph <N> <prompt>  (shorthand)\n\n"
                "Examples:\n"
                "  /ralph 20 Fix all type errors\n"
                "  /ralph Refactor auth --max 10 --done 'All tests pass'\n\n"
                "Commands:\n"
                "  /ralph-status - check progress\n"
                "  /ralph-cancel - stop loop"
            )
            return True

        if self.bot.processing:
            self.bot.send_reply("Already running. Use /ralph-cancel first.")
            return True

        self.bot.ralph_loop = RalphLoop(
            self.bot,
            ralph_args["prompt"],
            self.bot.working_dir,
            self.bot.output_dir,
            max_iterations=ralph_args["max_iterations"],
            completion_promise=ralph_args["completion_promise"],
            sessions=self.bot.sessions,
            ralph_loops=self.bot.ralph_loops,
        )
        self.bot.processing = True
        asyncio.ensure_future(cast(Awaitable[Any], self.bot.run_ralph()))
        return True
