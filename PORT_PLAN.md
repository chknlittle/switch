# XMPP -> OpenCode Bridge Port Plan

Goal: build a new repo for an OpenCode-backed XMPP bridge while preserving the current XMPP bridge.
Primary requirements:
- Stateless session store in the bridge (own history + summaries).
- Engine switching per message/session (OpenCode or Claude CLI Opus).
- Streaming output with low-verbosity default and optional verbose /peek-like view.
- New default agent: bridge (build mode), low verbosity, gpt-5.2-codex, normal reasoning.
- Skip permissions equivalent to --dangerously-skip-permissions.
- No subagents by default; prefer explicit /spawn to create sibling sessions.
- Separate dispatcher JID for OC bridge: oc@100.119.143.40 (do not reuse tx@...).
- Skills are tracked in-repo under .claude/skills and symlinked into ~/.claude/skills.

Sources used
- OpenCode server API overview: https://opencode.ai/docs/server/
- OpenCode event types and parts: https://github.com/anomalyco/opencode/blob/dev/packages/sdk/js/src/gen/types.gen.ts

-------------------------------------------------------------------------------
1) New repo layout (suggested)

/home/user/xmpp-opencode-bridge/
  PORT_PLAN.md
  README.md
  bridge.py
  utils.py
  sessions.db
  output/            # session output logs, optional
  locks/             # session lock files
  .opencode/
    opencode.json    # project config, agent + permissions
  .claude/
    skills/
      <skill-name>/SKILL.md

Notes
- Keep repo independent to avoid regression in existing xmpp repo.
- Reuse current XMPP bot logic, Telegram mirroring, and calendar wiring where needed.
- Replace tmux menu with a simple log tail pane for each session (tail full stream: text + tool progress + stats).

-------------------------------------------------------------------------------
2) OpenCode server usage (single instance)

Run server once (systemd or manual):
  opencode serve --port 4096 --hostname 127.0.0.1

Optional auth:
  OPENCODE_SERVER_PASSWORD=... OPENCODE_SERVER_USERNAME=opencode

Why single server?
- OpenCode supports multiple sessions via /session endpoints.
- Avoid port contention and multi-server state drift.

-------------------------------------------------------------------------------
3) Sessions mapping (stateless)

Each XMPP session maps to a bridge-owned session that stores history.
OpenCode sessions are optional and should be treated as stateless workers.

API (if OpenCode sessions are still used)
- POST /session -> returns Session (id)
- GET /session/:id -> validate session exists
- POST /session/:id/abort -> cancel

DB schema updates (new repo only)
- sessions table: add active_engine, model_id, reasoning_mode
- new table: session_messages (role, content, engine, timestamps)
- optional summary table: session_summaries (rolling summaries per session)

Suggested columns
  name TEXT PRIMARY KEY
  xmpp_jid TEXT UNIQUE NOT NULL
  xmpp_password TEXT NOT NULL
  active_engine TEXT DEFAULT 'opencode'
  model_id TEXT DEFAULT 'opencode/gpt-5.2-codex'
  reasoning_mode TEXT DEFAULT 'normal'
  tmux_name TEXT
  created_at TEXT NOT NULL
  last_active TEXT NOT NULL
  status TEXT DEFAULT 'active'

-------------------------------------------------------------------------------
4) Agent and permissions configuration

Goal: default agent "bridge" with build mode and low verbosity.

opencode.json (project)
  agent.bridge:
    mode: primary
    description: "XMPP bridge agent"
    model: "openrouter/gpt-5.2-codex"  # OpenCode engine
    tools: { skill: true, bash: true, edit: true, read: true, write: true }
    maxSteps: (optional) keep low to reduce verbose chains
    prompt: (optional) minimal system guidance for low verbosity
    options: { reasoning: "normal" }

Agent mode policy
- No subagents by default. Prefer explicit /spawn to create a sibling XMPP session.
- If subtask parts appear, ignore unless user asked for /spawn.

Permissions (skip prompts)
  permission.edit: allow
  permission.bash: allow
  permission.webfetch: allow
  permission.external_directory: allow
  permission.doom_loop: allow

Reasoning toggle
- Provide a bridge command to switch reasoning mode (normal/high) by updating per-message options
- If OpenCode exposes only per-agent config, update agent options in-memory for message payload

Model ID source
- OpenCode Zen docs specify model id format opencode/<model-id>.
- For GPT 5.2 Codex use opencode/gpt-5.2-codex.

-------------------------------------------------------------------------------
5) Streaming behavior (default low verbosity)

Use prompt_async + SSE (OpenCode engine):
- POST /session/:id/prompt_async
- GET /event (SSE)

Event types to handle
- message.part.updated: stream delta text
- message.updated: finalize, collect stats
- message.part.removed/message.removed: cleanup
- session.status: busy/idle
- session.error: show error
- permission.updated: ignored if permissions are allow; else reply

Streaming UX policy (default)
- Buffer deltas and emit every N chars or every T seconds (e.g. 160 chars or 1.5s)
- Avoid token-by-token spam
- Only stream text parts; ignore reasoning parts unless verbose mode enabled

Verbose mode (/peek or /stream-verbose)
- Stream all deltas immediately
- Include reasoning parts if present
- Include tool state transitions as progress messages

Streaming throttle policy
- Use combined size+time throttle for smooth mobile UX.

SSE reconnection policy
- If SSE is silent for 60s, reconnect to /event without aborting the run.
- If total stall exceeds 10 minutes, abort the run and report failure.

-------------------------------------------------------------------------------
6) Tool progress UX (Option A)

Tool parts are emitted via message.part.updated with part.type == "tool".

Map tool state to XMPP progress messages:
- pending -> "... [tool queued]"
- running -> "... [tool running: <title>]"
- completed -> "... [tool done: <title>]"
- error -> "... [tool failed: <error>]"

Throttle to avoid spam
- Default: send tool progress only on running/completed/error
- In verbose mode: send all state transitions

-------------------------------------------------------------------------------
7) Stats and final message format

OpenCode provides usage on step-finish and in assistant message metadata:
- tokens: input/output/reasoning/cache
- cost
- finish reason

Recommended final footer
  [tokens in/out reason cache | cost | duration]

Logging policy
- Cost/tokens footer is appended only in the bridge->XMPP output.
- Do not add stats to OpenCode context or stored message parts.

-------------------------------------------------------------------------------
8) Skill migration

OpenCode supports Claude-compatible skills but expects:
  .claude/skills/<name>/SKILL.md

Migration steps
- Move existing .claude/skills/*.md into folders
- Rename each file to SKILL.md
- Ensure YAML frontmatter has name + description

-------------------------------------------------------------------------------
9) Bridge command set (proposed)

- /peek [N]         -> enable verbose streaming for next reply or show last N lines
- /thinking high|normal -> toggle reasoning mode (OpenCode only)
- /agent oc|cc          -> switch engine (oc => OpenCode, cc => Claude CLI Opus 4.5)
- /stream on|off    -> enable/disable streaming in session
- /cancel           -> abort current run
- /spawn <prompt>   -> create sibling session and forward prompt
- /reset            -> new OpenCode session ID (fresh state)

Keep existing dispatcher commands
- /list /recent /kill <name> /cal /tg /help

-------------------------------------------------------------------------------
10) Legacy expectations to preserve (~/.xmpp)

- Output logs per session in output/ and a /peek view into recent output.
- /cancel to stop an active run, and /kill to close a session.
- !<command> for running shell commands and echoing output back.
- Sibling sessions via +prompt (or /spawn) to avoid blocking.
- Telegram mirroring and typing indicators when engaged (if enabled).
- Calendar integration commands (/cal) when available.
- Activity/history logging for session tracking.

Questions
5) Confirm command names: /peek, /reason, /stream, /stats, /reset?

-------------------------------------------------------------------------------
11) Minimal implementation checklist

1. Create new repo with existing XMPP logic (copy base from xmpp repo)
2. Add stateless session store (messages + summaries)
3. Implement OpenCodeRunner (stateless calls)
4. Implement ClaudeRunner Opus (stateless calls)
5. Add engine switch logic (/model codex|opus)
6. Implement SSE listener and per-session event routing
7. Implement streaming buffer and verbose toggle
8. Add tool progress notifications
9. Add reasoning toggle command (per message)
10. Update skill folder layout
11. Replace tmux session-shell menu with log tailer

-------------------------------------------------------------------------------
Open Questions Summary
1) Confirm if /thinking should be a no-op when agent is cc.
