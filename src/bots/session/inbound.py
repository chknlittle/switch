"""Inbound message parsing helpers for SessionBot."""

from __future__ import annotations

import json
import re


_URL_RE = re.compile(r"https?://[^\s<>\]\)\}]+", re.IGNORECASE)


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
                if tag.endswith("}url") or tag == "url":
                    text = (getattr(child, "text", None) or "").strip()
                    if text.startswith("http"):
                        urls.append(text)
    except Exception:
        pass

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
