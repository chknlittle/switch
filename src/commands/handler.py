"""CommandHandler composes mixins and dispatches slash commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast

from src.commands.mixins import (
    EngineCommandsMixin,
    HistoryCommandsMixin,
    MiscCommandsMixin,
    RalphCommandsMixin,
    SessionCommandsMixin,
)

if TYPE_CHECKING:
    from src.bots.session import SessionBot


class CommandHandler(
    SessionCommandsMixin,
    EngineCommandsMixin,
    RalphCommandsMixin,
    HistoryCommandsMixin,
    MiscCommandsMixin,
):
    """Handles slash commands for a session bot.

    Commands are registered via the @command decorator on mixin methods.
    The handler auto-discovers all decorated methods on init (walks MRO).
    """

    def __init__(self, bot: SessionBot):
        self.bot = bot
        self._commands: dict[str, tuple[Callable[..., Awaitable[bool]], bool]] = {}
        self._discover_commands()

    def _discover_commands(self) -> None:
        """Find all @command decorated methods on the MRO and register them."""
        for cls in type(self).__mro__:
            if cls is object:
                continue
            for attr in vars(cls).values():
                if not callable(attr) or not hasattr(attr, "_command_name"):
                    continue
                m = cast(Any, attr)
                cmd_name = cast(str, m._command_name)
                exact = cast(bool, m._command_exact)
                handler = cast(Callable[..., Awaitable[bool]], attr)
                self._commands[cmd_name] = (handler, exact)
                for alias in cast(tuple[str, ...], m._command_aliases):
                    self._commands[alias] = (handler, exact)

    async def handle(self, body: str) -> bool:
        """Handle a command. Returns True if command was handled."""
        cmd = body.strip().lower()

        # Exact matches first.
        for prefix, (handler, exact) in self._commands.items():
            if exact and cmd == prefix:
                return await handler(self, body)

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
            return await best[1](self, body)

        return False
