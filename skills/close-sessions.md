---
name: close-sessions
description: Use when user asks to close sessions, clean up stale sessions, kill old sessions, or manage XMPP bridge sessions. Triggers on "close session", "stale session", "cleanup sessions", "kill session", or session management requests.
version: 2.0.0
---

# Closing XMPP Bridge Sessions

## The Right Tool

Use the close-session.py script in ~/switch/scripts:

```bash
cd ~/switch && python scripts/close-session.py <session-name> [optional-message]
```

This script properly:
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
cd ~/switch && python scripts/close-session.py session-name-here
```

### Close with custom message
```bash
cd ~/switch && python scripts/close-session.py old-session "Cleaning up old sessions. Start a new chat!"
```

### Clean up multiple stale sessions
List first, then close each:
```bash
~/switch/scripts/sessions.sh list
cd ~/switch && python scripts/close-session.py stale-session-1
cd ~/switch && python scripts/close-session.py stale-session-2
```

## CRITICAL: Never Close Your Own Session

Do NOT close the session you are currently running in. Check your tmux session name first if unsure.

The user can identify stale sessions by:
- Old last_active timestamps
- Sessions they don't recognize
- Sessions that should have ended

## Alternative: sessions.sh kill

There's also `~/switch/scripts/sessions.sh kill <name>` but it:
- Does NOT send XMPP goodbye message
- DELETES the session from DB (vs marking closed)
- Less graceful

Prefer close-session.py for proper cleanup.
