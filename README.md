# XMPP-OpenCode Bridge

Chat with AI coding assistants from any XMPP client. Each conversation becomes a separate contact, making it easy to manage multiple concurrent sessions from your phone or desktop.

## Features

- **Multi-session**: Each conversation is a separate XMPP contact
- **Multiple backends**: Switch between OpenCode and Claude Code
- **Mobile-friendly**: Works with any XMPP client (Conversations, Gajim, Dino, etc.)
- **Session persistence**: Resume conversations after restarts
- **Ralph loops**: Autonomous iteration for long-running tasks
- **Shell access**: Run commands directly from chat

## Quick Start

```bash
# Install dependencies
uv sync

# Configure
cp .env.example .env
# Edit .env with your XMPP server details

# Run
uv run python bridge.py
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

- [Setup Guide](docs/setup.md) - Installation and configuration
- [Commands Reference](docs/commands.md) - All available commands
- [Architecture](docs/architecture.md) - How the system works

## Requirements

- Python 3.11+
- ejabberd XMPP server
- OpenCode CLI and/or Claude Code CLI
- tmux

## License

MIT
