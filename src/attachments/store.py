from __future__ import annotations

import hashlib
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import aiohttp

from .config import AttachmentsConfig, get_attachments_config


@dataclass(frozen=True)
class Attachment:
    id: str
    kind: str  # e.g. "image"
    mime: str
    filename: str
    local_path: str
    size_bytes: int
    sha256: str
    original_url: str | None = None
    public_url: str | None = None


_IMAGE_EXT_BY_MIME: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}


def _safe_slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "file"


def _guess_ext(mime: str, url: str | None = None) -> str:
    if mime in _IMAGE_EXT_BY_MIME:
        return _IMAGE_EXT_BY_MIME[mime]
    if url:
        path = urlparse(url).path
        _, ext = os.path.splitext(path)
        if ext and re.match(r"^\.[a-zA-Z0-9]{1,6}$", ext):
            return ext.lower()
    return ".bin"


def _is_disallowed_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return True
    if parsed.scheme not in {"http", "https"}:
        return True
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return True
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    # Best-effort guardrails against obvious private IP literals.
    if re.match(r"^10\.", host):
        return True
    if re.match(r"^192\.168\.", host):
        return True
    if re.match(r"^172\.(1[6-9]|2\d|3[0-1])\.", host):
        return True
    if host.startswith("169.254."):
        return True
    return False


class AttachmentStore:
    def __init__(
        self,
        base_dir: Path | None = None,
        public_base_url: str | None = None,
        token: str | None = None,
    ):
        cfg: AttachmentsConfig = get_attachments_config()

        self.base_dir = base_dir or cfg.base_dir
        resolved_public = public_base_url if public_base_url is not None else cfg.public_base_url
        self.public_base_url = (resolved_public or "").rstrip("/")
        self.token = token if token is not None else cfg.token

        self.base_dir.mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_name: str) -> Path:
        d = self.base_dir / _safe_slug(session_name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def build_public_url(self, session_name: str, filename: str) -> str | None:
        if not self.public_base_url or not self.token:
            return None
        sess = _safe_slug(session_name)
        return f"{self.public_base_url}/attachments/{self.token}/{sess}/{filename}"

    def _batch_dir(self, session_name: str) -> tuple[Path, str]:
        """Return (dir_path, rel_prefix) for a single message's attachments."""
        ts = int(time.time() * 1000)
        prefix = f"att_{ts}_{uuid.uuid4().hex[:6]}"
        local_dir = self.session_dir(session_name) / prefix
        local_dir.mkdir(parents=True, exist_ok=True)
        return local_dir, prefix

    async def download_images(self, session_name: str, urls: Iterable[str]) -> list[Attachment]:
        out: list[Attachment] = []
        max_bytes = int(os.getenv("SWITCH_ATTACHMENT_MAX_BYTES", str(10 * 1024 * 1024)))
        timeout_s = float(os.getenv("SWITCH_ATTACHMENT_FETCH_TIMEOUT_S", "20"))

        batch_dir: Path | None = None
        batch_prefix: str | None = None
        idx = 0

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as session:
            for url in urls:
                url = (url or "").strip()
                if not url or _is_disallowed_url(url):
                    continue

                try:
                    async with session.get(url, headers={"Accept": "image/*"}) as resp:
                        if resp.status >= 400:
                            continue
                        mime = (
                            (resp.headers.get("Content-Type") or "")
                            .split(";", 1)[0]
                            .strip()
                            .lower()
                        )
                        if not mime.startswith("image/"):
                            continue

                        data = bytearray()
                        async for chunk in resp.content.iter_chunked(64 * 1024):
                            data.extend(chunk)
                            if len(data) > max_bytes:
                                data = bytearray()
                                break
                        if not data:
                            continue

                        sha = hashlib.sha256(data).hexdigest()
                        ext = _guess_ext(mime, url=url)

                        if batch_dir is None or batch_prefix is None:
                            batch_dir, batch_prefix = self._batch_dir(session_name)

                        idx += 1
                        stem = f"img_{idx:02d}_{uuid.uuid4().hex[:6]}"
                        filename = f"{batch_prefix}/{stem}{ext}"
                        path = batch_dir / f"{stem}{ext}"
                        path.write_bytes(bytes(data))

                        att_id = f"att_{uuid.uuid4().hex[:12]}"
                        public_url = self.build_public_url(session_name, filename)
                        out.append(
                            Attachment(
                                id=att_id,
                                kind="image",
                                mime=mime,
                                filename=filename,
                                local_path=str(path),
                                size_bytes=len(data),
                                sha256=sha,
                                original_url=url,
                                public_url=public_url,
                            )
                        )
                except Exception:
                    continue

        return out
