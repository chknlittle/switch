# Switch

Chat with AI coding assistants from any XMPP client. Each conversation becomes a separate contact, making it easy to manage multiple concurrent sessions from your phone or desktop.

Designed to run on a dedicated Linux machine (old laptop, mini PC, home server) so the AI has real system access to do useful work.

## Features

- **Multi-session**: Each conversation is a separate XMPP contact
- **Multiple backends**: Switch between OpenCode and Claude Code
- **Mobile-friendly**: Works with any XMPP client (Conversations, Gajim, Dino, etc.)
- **Session persistence**: Resume conversations after restarts
- **Ralph loops**: Autonomous iteration for long-running tasks
- **Shell access**: Run commands directly from chat
- **Local memory vault**: Gitignored notes under `memory/`

## Quick Start

```bash
# Install dependencies
uv sync

# Configure
cp .env.example .env
# Edit .env with your XMPP server details

# Set up agent instructions (shared between both backends)
vim ~/AGENTS.md
ln -s ~/AGENTS.md ~/CLAUDE.md

# Run
uv run python -m src.bridge
```

Send a message to `oc@your.server` to create your first session.

## How It Works

```
You ──▶ XMPP Client ──▶ Dispatcher Bot ──▶ Session Bot ──▶ AI Backend
                            │                   │
                            │                   ├── OpenCode CLI
                            │                   └── Claude Code CLI
                            │
                            └── Creates new session contacts dynamically
```

## Basic Usage

| Action | Command |
|--------|---------|
| New session (OpenCode) | Send message to `oc@...` |
| New session (Claude) | `@cc <message>` to dispatcher |
| Switch to Claude | `/agent cc` in session |
| Switch to OpenCode | `/agent oc` in session |
| Cancel current run | `/cancel` |
| Run shell command | `!git status` |
| List sessions | `/list` to dispatcher |

## Documentation

- [Setup Guide](docs/setup.md) - Hardware, installation, configuration
- [Commands Reference](docs/commands.md) - All available commands
- [Architecture](docs/architecture.md) - How the system works
- [Memory Vault](docs/memory.md) - Store local learnings and runbooks

## Requirements

- Dedicated Linux machine (bare metal preferred)
- Python 3.11+
- ejabberd XMPP server
- OpenCode CLI and/or Claude Code CLI
- tmux
- [Tailscale](https://tailscale.com/) (recommended for secure remote access)

## Recommended Models

- **OpenCode**: GLM 4.7-flash via OpenRouter - fast, cheap, good for iteration
- **Claude Code**: Uses Claude Opus by default

## License

MIT
