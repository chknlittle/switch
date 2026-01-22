# XMPP OpenCode Bridge

XMPP bridge that mirrors the existing xmpp bridge but can route sessions to
OpenCode or Claude CLI.

## Setup

1. Copy env

```bash
cp .env.example .env
```

2. Create the dispatcher account once

```bash
ssh user@your.xmpp.server "/path/to/ejabberdctl register tx-oc your.xmpp.server <password>"
```

3. Start the service

```bash
./start.sh
```

## Commands

- Send any message to `oc@<server>` to spawn a new session.
- `/agent oc|cc` switches engines (OpenCode vs Claude).
- `/model <id>` sets OpenCode model id (ex: `openai/gpt-5.2-codex`).
- `/thinking normal|high` sets OpenCode reasoning mode.
- `/reset` clears current engine session id.
- `/peek [N]` shows last output lines (min 100).
- `/cancel` aborts active run.
- `!<command>` runs a shell command.

## Tmux Sessions

Each session creates a tmux window that tails the session log. Attach with:

```bash
tmux attach -t <session-name>
```

## Skills

Skills are tracked in this repo under `.claude/skills/` so both OpenCode and
Claude sessions can share the same playbooks.

Recommended setup:

```bash
ln -s /home/user/xmpp-opencode-bridge/.claude/skills /home/user/.claude/skills
```
```
