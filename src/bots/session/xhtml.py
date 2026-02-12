from __future__ import annotations

import re

from slixmpp.xmlstream import ET

XHTML_IM_NS = "http://jabber.org/protocol/xhtml-im"
XHTML_NS = "http://www.w3.org/1999/xhtml"

_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_ORDERED_LIST_RE = re.compile(r"^\s*\d+[.)]\s+")


def build_xhtml_message(text: str) -> ET.Element | None:
    """Build an XHTML-IM payload for the provided message text.

    We preserve markdown-like structure (paragraphs, lists, fenced code, tables)
    so capable clients can render a readable rich message instead of raw markdown.
    """

    normalized = _normalize(text)
    if not normalized.strip():
        return None

    html = ET.Element(f"{{{XHTML_IM_NS}}}html")
    body = ET.SubElement(html, f"{{{XHTML_NS}}}body")

    for kind, payload in _parse_blocks(normalized):
        if kind == "code":
            pre = ET.SubElement(body, f"{{{XHTML_NS}}}pre")
            code = ET.SubElement(pre, f"{{{XHTML_NS}}}code")
            code.text = payload
            continue

        if kind == "table":
            headers, rows = payload
            _append_table(body, headers, rows)
            continue

        if kind == "ul":
            ul = ET.SubElement(body, f"{{{XHTML_NS}}}ul")
            for item in payload:
                li = ET.SubElement(ul, f"{{{XHTML_NS}}}li")
                li.text = item
            continue

        if kind == "ol":
            ol = ET.SubElement(body, f"{{{XHTML_NS}}}ol")
            for item in payload:
                li = ET.SubElement(ol, f"{{{XHTML_NS}}}li")
                li.text = item
            continue

        p = ET.SubElement(body, f"{{{XHTML_NS}}}p")
        _set_with_breaks(p, payload)

    return html


def _normalize(text: str) -> str:
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
    )


def _parse_blocks(text: str) -> list[tuple[str, object]]:
    lines = text.split("\n")
    out: list[tuple[str, object]] = []
    i = 0
    n = len(lines)

    while i < n:
        if not lines[i].strip():
            i += 1
            continue

        if lines[i].lstrip().startswith("```"):
            i += 1
            code_lines: list[str] = []
            while i < n and not lines[i].lstrip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < n and lines[i].lstrip().startswith("```"):
                i += 1
            out.append(("code", "\n".join(code_lines)))
            continue

        if i + 1 < n and _looks_like_table_row(lines[i]) and _is_table_separator(lines[i + 1]):
            headers = _parse_table_row(lines[i])
            i += 2
            rows: list[list[str]] = []
            while i < n and _looks_like_table_row(lines[i]) and lines[i].strip():
                rows.append(_parse_table_row(lines[i]))
                i += 1
            out.append(("table", (headers, rows)))
            continue

        if _is_unordered_list_item(lines[i]):
            items: list[str] = []
            while i < n and _is_unordered_list_item(lines[i]):
                items.append(_strip_unordered_marker(lines[i]))
                i += 1
            out.append(("ul", items))
            continue

        if _is_ordered_list_item(lines[i]):
            items = []
            while i < n and _is_ordered_list_item(lines[i]):
                items.append(_strip_ordered_marker(lines[i]))
                i += 1
            out.append(("ol", items))
            continue

        para_lines = [lines[i]]
        i += 1
        while i < n and lines[i].strip():
            if lines[i].lstrip().startswith("```"):
                break
            if i + 1 < n and _looks_like_table_row(lines[i]) and _is_table_separator(lines[i + 1]):
                break
            if _is_unordered_list_item(lines[i]) or _is_ordered_list_item(lines[i]):
                break
            para_lines.append(lines[i])
            i += 1
        out.append(("p", "\n".join(para_lines).strip()))

    return out


def _set_with_breaks(node: ET.Element, text: str) -> None:
    parts = text.split("\n")
    node.text = parts[0] if parts else ""
    for part in parts[1:]:
        br = ET.SubElement(node, f"{{{XHTML_NS}}}br")
        br.tail = part


def _append_table(parent: ET.Element, headers: list[str], rows: list[list[str]]) -> None:
    table = ET.SubElement(parent, f"{{{XHTML_NS}}}table")
    thead = ET.SubElement(table, f"{{{XHTML_NS}}}thead")
    tr_head = ET.SubElement(thead, f"{{{XHTML_NS}}}tr")
    for h in headers:
        th = ET.SubElement(tr_head, f"{{{XHTML_NS}}}th")
        th.text = h

    if rows:
        tbody = ET.SubElement(table, f"{{{XHTML_NS}}}tbody")
        for row in rows:
            tr = ET.SubElement(tbody, f"{{{XHTML_NS}}}tr")
            for idx in range(len(headers)):
                cell = row[idx] if idx < len(row) else ""
                td = ET.SubElement(tr, f"{{{XHTML_NS}}}td")
                td.text = cell


def _looks_like_table_row(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped:
        return False
    cells = _parse_table_row(stripped)
    return len(cells) >= 2


def _is_table_separator(line: str) -> bool:
    return bool(_TABLE_SEPARATOR_RE.match(line))


def _parse_table_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [part.strip() for part in s.split("|")]


def _is_unordered_list_item(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("- ") or stripped.startswith("* ") or stripped.startswith("+ ")


def _strip_unordered_marker(line: str) -> str:
    stripped = line.lstrip()
    return stripped[2:].strip() if len(stripped) >= 2 else stripped


def _is_ordered_list_item(line: str) -> bool:
    return bool(_ORDERED_LIST_RE.match(line))


def _strip_ordered_marker(line: str) -> str:
    return _ORDERED_LIST_RE.sub("", line, count=1).strip()
