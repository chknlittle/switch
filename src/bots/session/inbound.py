"""Inbound message parsing helpers for SessionBot."""

from __future__ import annotations

import base64
import json
import logging
import re

log = logging.getLogger(__name__)


_URL_RE = re.compile(r"https?://[^\s<>\]\)\}]+", re.IGNORECASE)
_BOB_NS = "urn:xmpp:bob"


def extract_switch_meta(
    msg, *, meta_ns: str
) -> tuple[str | None, dict[str, str] | None, object | None]:
    """Extract Switch message meta extension (best-effort)."""
    for child in getattr(msg, "xml", []) or []:
        if getattr(child, "tag", None) != f"{{{meta_ns}}}meta":
            continue
        attrs = dict(getattr(child, "attrib", {}) or {})
        meta_type = attrs.get("type")

        payload_obj: object | None = None
        payload = child.find(f"{{{meta_ns}}}payload")
        if payload is not None and (payload.get("format") or "").lower() == "json":
            raw = (payload.text or "").strip()
            if raw:
                payload_obj = json.loads(raw)

        return meta_type, attrs, payload_obj

    return None, None, None


def extract_attachment_urls(msg, body: str) -> list[str]:
    urls: list[str] = []

    # jabber:x:oob and similar: scan all descendants for <url> elements.
    try:
        for el in getattr(msg, "xml", []) or []:
            for child in list(el.iter()):
                tag = getattr(child, "tag", "")
                # XEP-0372 and other variants sometimes use attributes.
                for attr in ("uri", "href"):
                    v = (child.get(attr) or "").strip()
                    if v.startswith("http"):
                        urls.append(v)
                if tag.endswith("}url") or tag == "url":
                    text = (getattr(child, "text", None) or "").strip()
                    if text.startswith("http"):
                        urls.append(text)
    except Exception:
        log.debug("Failed to extract attachment URLs from stanza", exc_info=True)

    for m in _URL_RE.finditer(body or ""):
        raw = m.group(0)
        url = raw.rstrip(".,;:!?")
        if url.startswith("http"):
            urls.append(url)

    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def extract_bob_images(msg) -> list[tuple[str, bytes, str | None]]:
    """Extract inline image payloads (XEP-0231 Bits of Binary).

    Returns (mime, bytes, original_url).
    """

    out: list[tuple[str, bytes, str | None]] = []
    seen_cids: set[str] = set()

    try:
        for el in getattr(msg, "xml", []) or []:
            for child in list(el.iter()):
                if getattr(child, "tag", None) != f"{{{_BOB_NS}}}data":
                    continue

                mime = ((child.get("type") or "").strip().lower())
                if not mime.startswith("image/"):
                    continue

                cid = (child.get("cid") or "").strip()
                if cid and cid in seen_cids:
                    continue

                raw = (getattr(child, "text", None) or "").strip()
                if not raw:
                    continue

                try:
                    data = base64.b64decode(raw.encode("utf-8"), validate=False)
                except Exception:
                    continue
                if not data:
                    continue

                if cid:
                    seen_cids.add(cid)
                original = f"cid:{cid}" if cid else None
                out.append((mime, data, original))
    except Exception:
        log.debug("Failed to extract BOB images from stanza", exc_info=True)

    return out


def strip_urls_from_body(body: str, urls: list[str]) -> str:
    if not body:
        return body
    out = body
    for u in urls:
        out = out.replace(u, "")
    return " ".join(out.split()).strip()


def normalize_leading_at(body: str) -> str:
    body = (body or "").strip()
    if body.startswith("@"):  # convenience alias for slash commands
        return "/" + body[1:]
    return body
