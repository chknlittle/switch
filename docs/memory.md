# Memory Vault

The memory vault is a local, gitignored space for storing session learnings,
runbooks, and operational notes. It is designed for fast filesystem search
without polluting the repo history.

Location
- `memory/`
- Contents are ignored by git; only `.gitignore` and `.gitkeep` are tracked

Recommended structure
- Organize by content area to keep searches focused
- Example:
  - `memory/helius/`
  - `memory/ejabberd/`
  - `memory/moonshot/`

How to use
1. Create a new topic folder if needed under `memory/`.
2. Add a concise, searchable markdown file.
3. Prefer one topic per file; name it for what you would search.

Example
```bash
mkdir -p memory/infra
vim memory/infra/tailscale-ops.md
```

Notes
- The vault is local-only; don’t expect changes to be shared via git.
- If you want a shared, versioned record, move the file into `docs/` instead.
