---
name: close-sessions
description: Use when user asks to close sessions, clean up stale sessions, kill old sessions, or manage XMPP bridge sessions. Triggers on "close session", "stale session", "cleanup sessions", "kill session", or session management requests.
version: 2.0.0
---

# Closing XMPP Bridge Sessions

## The Right Tool

Use `sessions.sh kill` in ~/switch/scripts:

```bash
~/switch/scripts/sessions.sh kill <session-name>
```

This properly:
- Sends a goodbye message via XMPP to the user
- Deletes the XMPP account from ejabberd
- Kills the tmux session
- Marks the session as closed in the database

## Listing Sessions First

Before closing, check what sessions exist:

```bash
~/switch/scripts/sessions.sh list
```

This shows:
- Database sessions (name, JID, last active time)
- Active tmux sessions

## Common Workflows

### Close a specific stale session
```bash
~/switch/scripts/sessions.sh kill session-name-here
```

### Close with custom message
Not currently supported via `sessions.sh` (it uses a standard goodbye message).

### Clean up multiple stale sessions
List first, then close each:
```bash
~/switch/scripts/sessions.sh list
~/switch/scripts/sessions.sh kill stale-session-1
~/switch/scripts/sessions.sh kill stale-session-2
```

## CRITICAL: Never Close Your Own Session

Do NOT close the session you are currently running in. Check your tmux session name first if unsure.

The user can identify stale sessions by:
- Old last_active timestamps
- Sessions they don't recognize
- Sessions that should have ended

## Alternative: sessions.sh kill

`~/switch/scripts/sessions.sh kill <name>` is the preferred tool.

It attempts to request the kill via the in-bridge dispatcher first so the in-memory
bot can wind down cleanly; if that fails it falls back to offline cleanup.
