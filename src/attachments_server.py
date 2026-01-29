from __future__ import annotations

import re
from pathlib import Path

from aiohttp import web


def _safe_part(text: str) -> str:
    out: list[str] = []
    for ch in (text or ""):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
    return "".join(out) or "_"


def _safe_relpath(text: str) -> str:
    """Sanitize a possibly-nested relative path.

    Allows path separators so attachments can be organized into subfolders,
    while preventing traversal and stripping unsafe characters.
    """

    raw = (text or "").strip()
    if not raw:
        return "_"

    raw = raw.replace("\\", "/")
    raw = re.sub(r"/+", "/", raw)

    parts: list[str] = []
    for part in raw.split("/"):
        part = part.strip()
        if not part or part in {".", ".."}:
            continue
        parts.append(_safe_part(part))

    # Keep it bounded to avoid abusive paths.
    if not parts:
        return "_"
    return "/".join(parts[:12])


async def start_attachments_server(
    base_dir: Path,
    *,
    token: str,
    host: str = "127.0.0.1",
    port: int = 7777,
) -> tuple[web.AppRunner, str, int]:
    """Start a tiny HTTP server to serve attachments.

    Exposes: /attachments/{token}/{session}/{path}
    """
    token = (token or "").strip()
    if not token:
        raise RuntimeError("Attachments server requires a token")

    app = web.Application()

    async def handle(request: web.Request) -> web.StreamResponse:
        req_token = request.match_info.get("token", "")
        if req_token != token:
            raise web.HTTPNotFound()

        sess = _safe_part(request.match_info.get("session", ""))
        rel = _safe_relpath(request.match_info.get("path", ""))
        path = (base_dir / sess / rel).resolve()
        base = base_dir.resolve()
        if base not in path.parents:
            raise web.HTTPNotFound()
        if not path.exists() or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    # Allow nested file paths so attachments can be grouped in subfolders.
    app.router.add_get("/attachments/{token}/{session}/{path:.*}", handle)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    return runner, host, port
