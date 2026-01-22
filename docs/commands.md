# Commands Reference

## Dispatcher Commands

Send these to the dispatcher bot (`oc@domain`):

| Command | Description |
|---------|-------------|
| `<message>` | Create new session with OpenCode (default) |
| `@cc <message>` | Create new session with Claude |
| `@oc <message>` | Create new session with OpenCode |
| `/list` | Show all sessions |
| `/recent` | Show 10 most recent sessions with status |
| `/kill <name>` | End a session and delete XMPP account |
| `/help` | Show help message |

## Session Commands

Send these to a session contact (`session-name@domain`):

### Engine Control

| Command | Description |
|---------|-------------|
| `/agent oc` | Switch to OpenCode backend |
| `/agent cc` | Switch to Claude backend |
| `/model <id>` | Set OpenCode model (e.g., `openai/gpt-4o`) |
| `/thinking normal\|high` | Set OpenCode reasoning mode |
| `/reset` | Clear session context (start fresh) |

### Execution Control

| Command | Description |
|---------|-------------|
| `/cancel` | Abort current AI run |
| `/peek [N]` | Show last N lines of output (default: 30, min: 100) |
| `/kill` | End this session |

### Shell Commands

| Command | Description |
|---------|-------------|
| `!<command>` | Run shell command and show output |

Example: `!git status`, `!pwd`, `!ls -la`

### Ralph Loop (Autonomous)

| Command | Description |
|---------|-------------|
| `/ralph <prompt>` | Start infinite loop (dangerous!) |
| `/ralph <N> <prompt>` | Run up to N iterations |
| `/ralph <prompt> --max N` | Same as above |
| `/ralph <prompt> --done "promise"` | Stop when AI outputs `<promise>...</promise>` |
| `/ralph-status` | Check loop progress |
| `/ralph-cancel` | Stop after current iteration |

Example:
```
/ralph 10 Fix all TypeScript errors --done "All errors fixed"
```

### Sibling Sessions

When a session is busy processing, prefix with `+` to spawn a parallel session:

```
+Start a new task while the other one runs
```

## Optional Integrations

### Calendar (`/cal`)

| Command | Description |
|---------|-------------|
| `/cal` | List upcoming events |
| `/cal add <title> <YYYY-MM-DD> [HH:MM]` | Add event |
| `/cal rm <event-id>` | Remove event |

### Telegram (`/tg`)

| Command | Description |
|---------|-------------|
| `/tg send <message>` | Send to all configured chats |
| `/tg history [N]` | Show last N messages |
| `/tg poll` | Fetch new messages |
