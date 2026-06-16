#!/bin/bash
# Reap stale Switch sessions.
#
# Heuristics (applied in order):
#   1. Dead pane   — tmux pane process has exited               → kill immediately
#   2. Orphaned    — tmux session has no matching DB entry       → kill immediately
#   3. Age > 3d    — tmux session created more than 3 days ago  → kill + close in DB
#   4. Stale > 48h — DB last_active older than 48 hours         → kill + close in DB
#
# Safe sessions: the current session's tmux name (if $TMUX_PANE is set) and any
# names passed via --keep.  The "switch" server session is always kept.
#
# Use --switch-only for cron jobs: only DB-tracked Switch sessions are eligible,
# and unrelated/orphan tmux sessions (dashboards, experiments, tunnels) are left alone.
#
# Usage:
#   reap-sessions.sh [--keep name1,name2] [--switch-only] [--dry-run]

set -euo pipefail
cd "$(dirname "$0")/.."

DB="sessions.db"
DRY_RUN=0
KEEP_CSV=""
SWITCH_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)     DRY_RUN=1; shift ;;
        --switch-only) SWITCH_ONLY=1; shift ;;
        --keep)        KEEP_CSV="$2"; shift 2 ;;
        *)             echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Build keep-set: always keep "switch" and our own session
declare -A KEEP
KEEP[switch]=1
if [ -n "${TMUX_PANE:-}" ]; then
    OWN=$(tmux display-message -p '#{session_name}' 2>/dev/null || true)
    [ -n "$OWN" ] && KEEP["$OWN"]=1
fi
IFS=',' read -ra EXTRAS <<< "$KEEP_CSV"
for k in "${EXTRAS[@]}"; do
    [ -n "$k" ] && KEEP["$k"]=1
done

is_kept() { [[ -v KEEP["$1"] ]]; }

KILLED=0

kill_tmux() {
    local name="$1" reason="$2"
    if (( DRY_RUN )); then
        echo "[dry-run] would kill tmux '$name' ($reason)"
    else
        tmux kill-session -t "$name" 2>/dev/null || true
        echo "killed tmux '$name' ($reason)"
    fi
    ((KILLED++)) || true
}

close_in_db() {
    local name="$1"
    (( DRY_RUN )) && return
    python3 - "$DB" "$name" <<'PY'
import sqlite3, sys, os
db, name = sys.argv[1], sys.argv[2]
conn = sqlite3.connect(db)
jid_row = conn.execute("SELECT xmpp_jid FROM sessions WHERE name = ?", (name,)).fetchone()
if jid_row and jid_row[0]:
    username = jid_row[0].split("@")[0]
    ejctl = os.environ.get("EJABBERD_CTL", "")
    domain = os.environ.get("XMPP_DOMAIN", "")
    if ejctl and domain:
        os.system(f'{ejctl} unregister {username} {domain} 2>/dev/null')
conn.execute("UPDATE sessions SET status='closed' WHERE name = ?", (name,))
conn.commit()
conn.close()
PY
}

source .env 2>/dev/null || true

NOW=$(date +%s)
THREE_DAYS=$((3 * 86400))

# Collect all tmux sessions
declare -A TMUX_SESSIONS  # name -> created_epoch
while IFS= read -r line; do
    name=$(echo "$line" | cut -d'|' -f1)
    created=$(echo "$line" | cut -d'|' -f2)
    TMUX_SESSIONS["$name"]="$created"
done < <(tmux list-sessions -F '#{session_name}|#{session_created}' 2>/dev/null || true)

# Collect DB active sessions with last_active
declare -A DB_ACTIVE  # name -> last_active_epoch
DB_QUERY_OK=0
DB_ACTIVE_COUNT=0

while IFS='|' read -r name epoch; do
    [ -n "$name" ] && DB_ACTIVE["$name"]="$epoch" && ((DB_ACTIVE_COUNT++))
done < <(python3 - "$DB" <<'PY'
import sqlite3, sys
from datetime import datetime, timezone
conn = sqlite3.connect(sys.argv[1])
for name, ts in conn.execute("SELECT name, last_active FROM sessions WHERE status = 'active'"):
    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            epoch = int(dt.timestamp())
        except Exception:
            epoch = 0
    else:
        epoch = 0
    print(f"{name}|{epoch}")
conn.close()
PY
) && DB_QUERY_OK=1

# Safety check: only proceed with orphaned killing if DB query succeeded and has entries
if (( DB_QUERY_OK )) && (( DB_ACTIVE_COUNT == 0 )); then
    echo "Warning: DB query returned 0 active sessions, skipping orphaned check to avoid killing valid sessions"
fi

# --- Pass 1: Dead panes ---
for name in "${!TMUX_SESSIONS[@]}"; do
    is_kept "$name" && continue
    (( SWITCH_ONLY )) && [[ ! -v DB_ACTIVE["$name"] ]] && continue
    # Check if pane command is still alive
    pid=$(tmux list-panes -t "$name" -F '#{pane_pid}' 2>/dev/null | head -1)
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
        kill_tmux "$name" "dead pane"
        [[ -v DB_ACTIVE["$name"] ]] && close_in_db "$name"
        unset "TMUX_SESSIONS[$name]"
    fi
done

# --- Pass 2: Orphaned tmux (no DB entry) ---
# Only run this pass if DB query succeeded AND we have entries
# This prevents killing non-Switch tmux sessions when DB is unreachable
if (( ! SWITCH_ONLY )) && (( DB_QUERY_OK )) && (( DB_ACTIVE_COUNT > 0 )); then
    for name in "${!TMUX_SESSIONS[@]}"; do
        is_kept "$name" && continue
        if [[ ! -v DB_ACTIVE["$name"] ]]; then
            kill_tmux "$name" "orphaned, no DB entry"
            unset "TMUX_SESSIONS[$name]"
        fi
    done
fi

# --- Pass 3: Age > 3 days (Switch sessions only) ---
# Only kill old sessions if they're tracked in DB (Switch sessions)
# Non-Switch tmux sessions are left alone regardless of age
for name in "${!TMUX_SESSIONS[@]}"; do
    is_kept "$name" && continue
    # Only apply age limit to Switch sessions (those in DB)
    [[ ! -v DB_ACTIVE["$name"] ]] && continue
    created="${TMUX_SESSIONS[$name]}"
    age=$(( NOW - created ))
    if (( age > THREE_DAYS )); then
        days=$(( age / 86400 ))
        kill_tmux "$name" "age: ${days}d"
        close_in_db "$name"
        unset "TMUX_SESSIONS[$name]"
    fi
done

# --- Pass 4: DB stale > 48h ---
FORTY_EIGHT_H=$((48 * 3600))
for name in "${!DB_ACTIVE[@]}"; do
    is_kept "$name" && continue
    [[ ! -v TMUX_SESSIONS["$name"] ]] && continue  # already handled or no tmux
    last="${DB_ACTIVE[$name]}"
    idle=$(( NOW - last ))
    if (( idle > FORTY_EIGHT_H )); then
        hours=$(( idle / 3600 ))
        kill_tmux "$name" "stale: ${hours}h since last_active"
        close_in_db "$name"
        unset "TMUX_SESSIONS[$name]"
    fi
done

echo "Reaped $KILLED session(s). $(( ${#TMUX_SESSIONS[@]} )) remaining."
