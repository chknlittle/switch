#!/usr/bin/env bash
# Offline smoke harness — run from repo root after refactors.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "FAIL  Missing venv at ${VENV_PYTHON} — run: uv sync" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "$VENV_PYTHON" "${REPO_ROOT}/scripts/smoke.py" "$@"
