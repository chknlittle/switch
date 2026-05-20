"""Shared helpers for command mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.bots.session import SessionBot


class CommandMixinBase:
    """Base mixin with shared message helpers."""

    bot: SessionBot

    def _recent_messages(
        self, session_name: str | None = None, limit: int = 10
    ) -> list[Any]:
        target_session = session_name or self.bot.session_name
        return self.bot.messages.list_recent(target_session, limit=limit)

    def _latest_message(
        self, role: str, *, session_name: str | None = None, limit: int = 10
    ) -> Any | None:
        for msg in self._recent_messages(session_name=session_name, limit=limit):
            if msg.role == role and msg.content.strip():
                return msg
        return None

    @staticmethod
    def _truncate_content(content: str, limit: int) -> str:
        if len(content) <= limit:
            return content
        return content[:limit] + "..."

    def _format_message_lines(
        self, messages: list[Any], *, truncate_at: int
    ) -> list[str]:
        lines: list[str] = []
        for msg in messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            lines.append(
                f"{prefix}: {self._truncate_content(msg.content, truncate_at)}"
            )
        return lines
