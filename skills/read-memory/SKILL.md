---
name: read-memory
description: Use when user asks about memory, recalls, past learnings, "what do we know about X", "@memory", "check memory", or when starting work on a topic that might have prior context stored. Also triggers on "remember", "recall", "did we", "have we".
version: 1.0.0
---

# Reading Memory

The memory vault stores learnings, runbooks, and operational notes from past sessions.

## Location

    ~/switch/memory/

## Structure

Memory is organized by topic:

    memory/
    ├── ejabberd/
    ├── git/
    ├── helius/
    ├── moonshot/
    ├── pumpswap/
    ├── python/
    └── telegram/

## How to Search

### List all topics

```bash
ls ~/switch/memory/
```

### List files in a topic

```bash
ls ~/switch/memory/helius/
```

### Search across all memory

```bash
grep -r "search term" ~/switch/memory/
```

### Read a specific memory file

```bash
cat ~/switch/memory/git/git-index-slowness-fix.md
```

## When to Read Memory

- At session start for relevant topics
- When user asks "what do we know about X"
- Before tackling a problem that might have prior solutions
- When user references past work or discoveries

## Important

- Memory is local-only (gitignored)
- If a memory file is useful for others, suggest moving it to `docs/`
- Memory files are concise markdown - scan them quickly
