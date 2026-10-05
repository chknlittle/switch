---
name: write-memory
description: Use when user wants to save learnings, add to memory, persist discoveries, "remember this", "save to memory", "@memory add", or when a session produces reusable knowledge that should be preserved. Triggers on "save", "store", "persist", "remember", "add to memory".
version: 1.0.0
---

# Writing to Memory

Save discoveries, fixes, and learnings to the memory vault for future sessions.

## Location

    ~/switch/memory/

## When to Write Memory

- Session produced a reusable fix or solution
- Discovered a gotcha or pattern that applies broadly
- Found useful commands or workflows
- User explicitly asks to save something
- Before spawning a new session (capture discoveries)

## How to Write

### 1. Choose or create a topic folder

```bash
# List existing topics
ls ~/switch/memory/

# Create new topic if needed
mkdir -p ~/switch/memory/new-topic/
```

### 2. Create a concise markdown file

Name it for what you would search:

```bash
# Good names (searchable)
memory/helius/websocket-keepalive.md
memory/git/git-index-slowness-fix.md
memory/python/style-preference.md

# Bad names (not searchable)
memory/notes.md
memory/stuff.md
```

### 3. Write concise, actionable content

Template:

```markdown
# Title

Symptoms (if a fix)
- What went wrong
- How to recognize it

Cause (if applicable)
- Root cause explanation

Fix/Solution
- Steps to resolve
- Commands to run

Notes
- Edge cases
- Related issues
```

## Example

```bash
cat > ~/switch/memory/infra/tailscale-reconnect.md << 'EOF'
# Tailscale Reconnect

Symptoms
- SSH to remote machines times out
- tailscale status shows "offline"

Fix
```bash
sudo tailscale down && sudo tailscale up
```

If that fails, restart the daemon:
```bash
sudo systemctl restart tailscaled
```
EOF
```

## Important

- One topic per file
- Keep files concise (under 50 lines ideal)
- Memory is local-only (gitignored) - won't be shared via git
- For shared knowledge, use `docs/` instead
