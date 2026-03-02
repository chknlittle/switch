"""Ports (interfaces) for runner implementations.

The rest of the system (bots, lifecycle, commands) should depend on these
contracts rather than concrete runner implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Protocol


RunnerEvent = tuple[str, object]


@dataclass
class Question:
    """A question from the AI to the user."""

    request_id: str
    questions: list[dict]  # [{header, question, options: [{label, description}]}]


# Type for question callback: receives Question, returns answers array
# Each answer is a list of selected option labels (one per question, positional)
QuestionCallback = Callable[[Question], Awaitable[list[list[str]]]]


class Runner(Protocol):
    """A streaming runner (engine adapter)."""

    def run(self, prompt: str, session_id: str | None = None) -> AsyncIterator[RunnerEvent]:
        ...

    def cancel(self) -> None:
        ...
