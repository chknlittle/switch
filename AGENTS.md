# Switch - AI Agent Guide

## What This Project Is

Switch is an XMPP bridge connecting AI coding assistants (Claude Code, OpenCode) to XMPP chat clients. It runs on a dedicated Linux machine and creates separate XMPP contacts for each AI conversation.

**Tech Stack**: Python 3.11+, slixmpp, SQLite, asyncio, ejabberd

## Environment

- Working directory is `~` (home)
- Switch repo lives at `~/switch`
- Use `uv run` to execute Python (e.g., `uv run python -m src.bridge`)
- Use `uv sync` to install dependencies
- Configuration lives in `.env` (see `.env.example`)
- Logs go to `~/switch/output/`
- Database is `~/switch/sessions.db` (SQLite)

## Code Conventions

### Commands

Commands use decorator-based registration. To add a new command:

```python
@command("/mycommand")
async def mycommand(self, body: str) -> bool:
    """Docstring describes the command."""
    self.bot.send_reply("Response")
    return True
```

- Use `@command("/name")` for exact match
- Use `@command("/name", exact=False)` for prefix match
- Use `@command("/name", "/alias")` for aliases

### Runners

Runners wrap CLI tools (Claude, OpenCode). They:
- Execute via subprocess with streaming JSON output
- Parse events and accumulate state
- Are cancellable via `runner.cancel()`

### Bots

- **DispatcherBot**: Listens on fixed JIDs, creates sessions on message
- **SessionBot**: One per conversation, routes messages to runners

### Database

Uses repository pattern. Tables defined in `src/db.py`. Access via repository classes, not raw SQL.

## Debugging

```bash
~/switch/scripts/logs.sh              # Real-time logs
sqlite3 ~/switch/sessions.db          # Query database
cat ~/switch/output/<session>.log     # Session-specific log
tmux attach -t <session>              # Attach to session tmux
```

## Memory

Project-specific memory lives in `~/switch/memory/` (gitignored). Use this for persistent notes, decisions, and context that should survive across sessions.

## Skills

Runbooks and reusable procedures live in `~/switch/skills/`. Reference these for common operations.
