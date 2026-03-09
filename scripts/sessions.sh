#!/bin/bash
# List or manage Switch sessions

cd "$(dirname "$0")/.."

DB="sessions.db"

has_sqlite() {
    command -v sqlite3 >/dev/null 2>&1
}

python_sqlite() {
    python3 - "$@" <<'PY'
import sqlite3
import sys

db = sys.argv[1]
mode = sys.argv[2]

conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row

if mode == "list":
    rows = conn.execute(
        "SELECT name, xmpp_jid, datetime(last_active) AS last_active FROM sessions ORDER BY last_active DESC"
    ).fetchall()
    if rows:
        print(f"{'name':<28} {'xmpp_jid':<40} last_active")
        for row in rows:
            print(f"{row['name']:<28} {row['xmpp_jid']:<40} {row['last_active'] or ''}")
elif mode == "jid":
    row = conn.execute("SELECT xmpp_jid FROM sessions WHERE name = ?", (sys.argv[3],)).fetchone()
    if row and row[0]:
        print(row[0])
elif mode == "close":
    conn.execute("UPDATE sessions SET status='closed' WHERE name = ?", (sys.argv[3],))
    conn.commit()

conn.close()
PY
}

require_sqlite() {
    if ! has_sqlite; then
        echo "sqlite3 not found; install it to manage sessions.db"
        exit 1
    fi
}

case "${1:-list}" in
    list)
        echo "=== Sessions ==="
        if [ -f "$DB" ]; then
            if has_sqlite; then
                sqlite3 -header -column "$DB" \
                    "SELECT name, xmpp_jid, datetime(last_active) as last_active FROM sessions ORDER BY last_active DESC"
            else
                python_sqlite "$DB" list || echo "Unable to read sessions.db"
            fi
        else
            echo "No sessions yet."
        fi
        echo ""
        echo "=== tmux Sessions ==="
        tmux list-sessions 2>/dev/null | grep -v "switch" || echo "None"
        ;;

    kill)
        if [ -z "$2" ]; then
            echo "Usage: $0 kill <session-name>"
            exit 1
        fi
        NAME="$2"

        # Do offline cleanup directly. Using scripts/spawn-session here wraps
        # /kill in a switch-spawn token, which creates another session instead
        # of dispatching the slash command.
        echo "Running offline cleanup."

        if has_sqlite; then
            JID=$(sqlite3 "$DB" "SELECT xmpp_jid FROM sessions WHERE name='$NAME'" 2>/dev/null)
        else
            JID=$(python_sqlite "$DB" jid "$NAME")
        fi
        if [ -z "$JID" ]; then
            echo "Session not found: $NAME"
            exit 1
        fi

        tmux kill-session -t "$NAME" 2>/dev/null

        USERNAME=$(echo "$JID" | cut -d@ -f1)
        source .env 2>/dev/null
        $EJABBERD_CTL unregister "$USERNAME" "$XMPP_DOMAIN" 2>/dev/null

        # Archive semantics: don't delete session history.
        if has_sqlite; then
            sqlite3 "$DB" "UPDATE sessions SET status='closed' WHERE name='$NAME'"
        else
            python_sqlite "$DB" close "$NAME"
        fi

        echo "Closed session: $NAME"
        ;;

    clean)
        echo "Cleaning all sessions..."
        if has_sqlite; then
            sqlite3 "$DB" "SELECT name FROM sessions" 2>/dev/null | while read NAME; do
                "$0" kill "$NAME"
            done
        else
            python3 - "$DB" <<'PY' | while read NAME; do
import sqlite3
import sys

conn = sqlite3.connect(sys.argv[1])
for (name,) in conn.execute("SELECT name FROM sessions"):
    print(name)
conn.close()
PY
                "$0" kill "$NAME"
            done
        fi
        echo "Done."
        ;;

    *)
        echo "Usage: $0 [list|kill <name>|clean]"
        ;;
esac
