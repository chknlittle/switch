"""Command registration decorator."""

from __future__ import annotations

from typing import Awaitable, Callable


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
