from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AttachmentsConfig:
    base_dir: Path
    token: str
    host: str
    port: int
    public_base_url: str


def _default_base_dir() -> Path:
    # Default to a stable path inside the switch repo so all agents can access it.
    default = Path(__file__).resolve().parents[2] / "uploads"
    return Path(os.getenv("SWITCH_ATTACHMENTS_DIR", str(default)))


def _load_or_create_token(base_dir: Path) -> str:
    env = (os.getenv("SWITCH_ATTACHMENTS_TOKEN") or "").strip()
    if env:
        return env

    token_path = base_dir / ".token"
    if token_path.exists():
        try:
            t = token_path.read_text(encoding="utf-8", errors="replace").strip()
            if t:
                return t
        except Exception:
            pass

    base_dir.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(24)
    try:
        token_path.write_text(token, encoding="utf-8")
        try:
            os.chmod(token_path, 0o600)
        except Exception:
            pass
    except Exception:
        # Best-effort: if we can't persist it, still return a token for this run.
        pass
    return token


def get_attachments_config() -> AttachmentsConfig:
    base_dir = _default_base_dir()
    host = (os.getenv("SWITCH_ATTACHMENTS_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("SWITCH_ATTACHMENTS_PORT", "7777"))
    token = _load_or_create_token(base_dir)

    public_base_url = (os.getenv("SWITCH_PUBLIC_ATTACHMENT_BASE_URL") or "").strip()
    if public_base_url:
        public_base_url = public_base_url.rstrip("/")
    else:
        url_host = host
        if url_host in {"0.0.0.0", "::", "[::]", ""}:
            url_host = "127.0.0.1"
        public_base_url = f"http://{url_host}:{port}"

    return AttachmentsConfig(
        base_dir=base_dir,
        token=token,
        host=host,
        port=port,
        public_base_url=public_base_url,
    )
