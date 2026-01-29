from __future__ import annotations

from pathlib import Path

from aiohttp import web


def _safe_part(text: str) -> str:
    out: list[str] = []
    for ch in (text or ""):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
    return "".join(out) or "_"


async def start_attachments_server(
    base_dir: Path,
    *,
    token: str,
    host: str = "127.0.0.1",
    port: int = 7777,
) -> tuple[web.AppRunner, str, int]:
    """Start a tiny HTTP server to serve attachments.

    Exposes: /attachments/{token}/{session}/{filename}
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
        name = _safe_part(request.match_info.get("filename", ""))
        path = (base_dir / sess / name).resolve()
        base = base_dir.resolve()
        if base not in path.parents:
            raise web.HTTPNotFound()
        if not path.exists() or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    app.router.add_get("/attachments/{token}/{session}/{filename}", handle)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    return runner, host, port
