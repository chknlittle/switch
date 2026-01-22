---
name: spawn-session
description: Use when user asks to spawn a new session, start fresh session, handoff to new session, create new session with summary, or continue work in a new context. Triggers on "spawn session", "new session", "fresh session", "handoff", "start new", "continue in new session".
version: 2.1.0
---

# Spawning a New XMPP Session

When context is getting large or the user wants to continue work in a fresh session, spawn a new session with a structured handoff.

## Before Spawning: Structured Handoff

Present this template to the user and fill it out together:

    HANDOFF SUMMARY
    ===============

    COMPLETED
    - [what was finished this session]
    - [include file paths where relevant]

    IN PROGRESS
    - [what was being worked on]
    - [current state, any blockers]

    NEXT STEPS
    1. [highest priority, dependency-ordered]
    2. [specific and actionable]
    3. [sized for one session each]

    DISCOVERIES
    - [patterns found that apply broadly]
    - [gotchas to avoid]
    - [useful commands/workflows]

    KEY FILES
    - path/to/start/reading.py
    - another/relevant/file.ts

After filling this out, ask:

    Save discoveries to @memory? (for future sessions)

If yes, use @memory to persist the discoveries before spawning.

## Script Location

    ~/switch/scripts/spawn-session.py

## Usage

    cd ~/switch && python scripts/spawn-session.py "<handoff message>"

## Formatting the Handoff Message

Combine the template into a single message for the new session:

    cd ~/switch && python scripts/spawn-session.py "HANDOFF FROM PREVIOUS SESSION

    COMPLETED
    - Implemented X in src/foo.py
    - Fixed bug in utils/bar.ts

    IN PROGRESS
    - Working on Y feature
    - Blocked on: need API credentials

    NEXT STEPS
    1. Get API creds from user
    2. Complete Y feature
    3. Add tests

    KEY FILES
    - src/foo.py
    - utils/bar.ts

    ---
    Continue from Next Steps. Read Key Files first."

## Quick Handoff (simple cases)

For straightforward handoffs without much context:

    cd ~/switch && python scripts/spawn-session.py "Continue [project].

    Done: [brief summary]
    Next: [what to do]
    Start with: [file or command]"

## Important Notes

- New session appears as new contact in Siskin
- Sessions default to OpenCode; switch with /agent cc if needed
- This session continues until explicitly closed
- User gets notified of new session
- ALWAYS capture discoveries before spawning - they're lost otherwise
