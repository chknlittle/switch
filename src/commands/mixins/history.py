"""Message history and context commands."""

from __future__ import annotations

from src.commands.mixins.base import CommandMixinBase
from src.commands.registry import command


class HistoryCommandsMixin(CommandMixinBase):
    @command("/last")
    async def last(self, _body: str) -> bool:
        """Show last assistant message."""
        msg = self._latest_message("assistant", limit=10)
        if msg:
            self.bot.send_reply(msg.content)
            return True
        self.bot.send_reply("No assistant messages in this session.")
        return True

    @command("/retry")
    async def retry(self, _body: str) -> bool:
        """Re-run last user prompt."""
        if self.bot.processing:
            self.bot.send_reply("Already processing. /cancel first, then /retry.")
            return True
        msg = self._latest_message("user", limit=20)
        if msg:
            self.bot.send_reply(f"Retrying: {self._truncate_content(msg.content, 80)}")
            await self.bot.session.enqueue(
                msg.content,
                None,
                trigger_response=True,
                scheduled=False,
                wait=False,
            )
            return True
        self.bot.send_reply("No user messages to retry.")
        return True

    @command("/recap")
    async def recap(self, _body: str) -> bool:
        """Summarize session history."""
        if self.bot.processing:
            self.bot.send_reply("Already processing. Try /recap after current work completes.")
            return True
        messages = self._recent_messages(limit=40)
        if not messages:
            self.bot.send_reply("No messages in this session.")
            return True

        # Chronological order, truncate long messages
        messages = list(reversed(messages))
        lines = self._format_message_lines(messages, truncate_at=500)

        recap_prompt = (
            "Summarize this conversation concisely. "
            "Key decisions, open questions, current status. Under 300 words.\n\n"
            "---\n" + "\n\n".join(lines) + "\n---"
        )
        self.bot.send_reply("Generating recap...")
        await self.bot.session.enqueue(
            recap_prompt, None,
            trigger_response=True, scheduled=False, wait=False,
        )
        return True

    @command("/context", exact=False)
    async def context(self, body: str) -> bool:
        """Inject cross-session history."""
        parts = body.strip().split()
        source_name = None
        limit = 20

        for part in parts[1:]:
            if part.startswith("from:"):
                source_name = part[5:]
            else:
                try:
                    limit = int(part)
                except ValueError:
                    pass

        if not source_name:
            self.bot.send_reply("Usage: /context from:<session-name> [limit]")
            return True

        source = self.bot.sessions.get(source_name)
        if not source:
            self.bot.send_reply(f"Session '{source_name}' not found.")
            return True

        messages = self._recent_messages(session_name=source_name, limit=limit)
        if not messages:
            self.bot.send_reply(f"No messages in '{source_name}'.")
            return True

        messages = list(reversed(messages))  # chronological
        lines = self._format_message_lines(messages, truncate_at=800)

        context_text = (
            f"[Context from session '{source_name}' — {len(messages)} messages. "
            "Use this as background for the conversation.]\n\n"
            + "\n\n".join(lines)
        )

        self.bot.session.set_context_prefix(context_text)
        self.bot.send_reply(
            f"Loaded {len(messages)} messages from '{source_name}'. "
            "Your next message will include this context."
        )
        return True
