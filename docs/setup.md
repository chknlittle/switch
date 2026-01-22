# Setup Guide

## Hardware

Switch is designed to run on a **dedicated Linux machine** - ideally bare metal so the AI agents have real system access and can do useful work. An old laptop, mini PC, or home server works well.

The machine should have:
- Full filesystem access for the AI to read/write code
- Ability to run compilers, tests, docker, etc.
- Network access for git, package managers, APIs

Running in a VM or container defeats the purpose - you want the AI to operate on a real development environment.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- ejabberd XMPP server (local or remote)
- tmux
- One of:
  - [OpenCode](https://github.com/opencode-ai/opencode) CLI
  - [Claude Code](https://claude.ai/code) CLI

## Installation

1. Clone the repository:

```bash
git clone <repo-url> switch
cd switch
```

2. Install dependencies:

```bash
uv sync
```

3. Copy and configure environment:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
XMPP_SERVER=your.xmpp.server
XMPP_DOMAIN=your.xmpp.server
XMPP_DISPATCHER_JID=oc@your.xmpp.server
XMPP_DISPATCHER_PASSWORD=your-dispatcher-password
XMPP_RECIPIENT=your-user@your.xmpp.server
EJABBERD_CTL=/path/to/ejabberdctl
```

## ejabberd Setup

### Create Dispatcher Account

The dispatcher bot needs a dedicated XMPP account:

```bash
ejabberdctl register tx-oc your.xmpp.server <password>
```

### Remote ejabberd

If ejabberd runs on a different machine, set `EJABBERD_CTL` to an SSH command:

```bash
EJABBERD_CTL="ssh user@host /path/to/ejabberdctl"
```

### Account Permissions

Switch creates/deletes XMPP accounts dynamically. Ensure ejabberd allows:
- Account registration via ejabberdctl
- Roster manipulation via ejabberdctl

## Agent Instructions

Both Claude Code and OpenCode look for instruction files in the working directory (`CLAUDE_WORKING_DIR`, defaults to `$HOME`).

- **OpenCode** reads `AGENTS.md`
- **Claude Code** reads `CLAUDE.md`

To share instructions between both backends, create `AGENTS.md` and symlink `CLAUDE.md` to it:

```bash
# Create your agent instructions
vim ~/AGENTS.md

# Symlink for Claude Code
ln -s ~/AGENTS.md ~/CLAUDE.md
```

## Running

### Direct

```bash
uv run python -m src.bridge
```

### As systemd Service

Copy the service file:

```bash
cp switch.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable switch
systemctl --user start switch
```

### Useful Commands

```bash
systemctl --user start switch    # Start service
systemctl --user stop switch     # Stop service
journalctl --user -u switch -f   # View logs
scripts/run.sh                   # Run directly (not via systemd)
```

## Verification

1. Start Switch
2. Send a message to `oc@your.xmpp.server` from your XMPP client
3. A new contact should appear with the session name
4. The AI should respond to your message

## Directory Structure

```
switch/
├── src/                # Main application
│   ├── bridge.py       # Entry point
│   └── utils.py        # XMPP utilities
├── scripts/            # Utility scripts
│   ├── run.sh          # Run directly (not via systemd)
│   ├── sessions.sh     # List/kill sessions
│   ├── spawn-session.py
│   └── close-session.py
├── docs/               # Documentation
├── pyproject.toml      # Dependencies
├── sessions.db         # SQLite database (created on first run)
├── output/             # Session output logs
└── .env                # Configuration (not committed)
```
