"""Markdown block parsing: split a document into heading, paragraph, list,
code and table blocks without touching inline syntax.

The chunker needs blocks it can treat as atoms -- a fenced code block or a
table row never gets sliced in half -- and it needs to know which heading
each block falls under. That is all this module does. Inline markdown
(emphasis, links, that sort of thing) is left untouched, because a chunker
never reads it.
"""

import re
from dataclasses import dataclass

__all__ = ["Block", "parse_blocks"]

_ATX_HEADING = re.compile(r"^(#{1,6})(?:\s+(.*?))?\s*#*\s*$")
_FENCE = re.compile(r"^(```+|~~~+)\s*([^\s`~]*)\s*$")
_ORDERED_ITEM = re.compile(r"^\s*\d+[.)]\s+")
_BULLET_ITEM = re.compile(r"^\s*[-*+]\s+")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")


@dataclass(frozen=True)
class Block:
    """One structural unit of a document.

    ``kind`` is one of ``heading``, ``paragraph``, ``list``, ``code`` or
    ``table``. ``level`` is set only for headings (1-6). ``start_line`` and
    ``end_line`` are 1-based and inclusive, pointing back at the source text.
    """

    kind: str
    text: str
    start_line: int
    end_line: int
    level: int = None


def parse_blocks(text):
    """Split ``text`` into a list of :class:`Block` in document order."""
    if not text:
        return []
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.strip() == "":
            i += 1
            continue
        heading = _ATX_HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            heading_text = (heading.group(2) or "").strip()
            blocks.append(Block("heading", heading_text, i + 1, i + 1, level))
            i += 1
            continue
        if _FENCE.match(line):
            block, i = _read_code_block(lines, i)
            blocks.append(block)
            continue
        if _looks_like_table_start(lines, i):
            block, i = _read_table(lines, i)
            blocks.append(block)
            continue
        if _BULLET_ITEM.match(line) or _ORDERED_ITEM.match(line):
            block, i = _read_list(lines, i)
            blocks.append(block)
            continue
        block, i = _read_paragraph(lines, i)
        blocks.append(block)
    return blocks


def _read_code_block(lines, start):
    """Consume a fenced code block. An unterminated fence runs to EOF."""
    fence = _FENCE.match(lines[start])
    marker = fence.group(1)[0]
    fence_len = len(fence.group(1))
    n = len(lines)
    i = start + 1
    while i < n:
        closing = _FENCE.match(lines[i])
        if (
            closing
            and closing.group(1)[0] == marker
            and len(closing.group(1)) >= fence_len
            and not closing.group(2)
        ):
            i += 1
            break
        i += 1
    text = "\n".join(lines[start:i])
    return Block("code", text, start + 1, i, None), i


def _looks_like_table_start(lines, i):
    if "|" not in lines[i]:
        return False
    if i + 1 >= len(lines):
        return False
    separator = lines[i + 1]
    return "-" in separator and bool(_TABLE_SEPARATOR.match(separator))


def _read_table(lines, start):
    n = len(lines)
    i = start + 1
    while i < n:
        line = lines[i]
        if line.strip() == "" or "|" not in line:
            break
        i += 1
    text = "\n".join(lines[start:i])
    return Block("table", text, start + 1, i, None), i


def _read_list(lines, start):
    """Consume a run of list items, allowing blank lines followed by another
    item or an indented continuation. A blank line followed by anything else
    ends the list."""
    n = len(lines)
    i = start
    while i < n:
        line = lines[i]
        if line.strip() == "":
            following = lines[i + 1] if i + 1 < n else ""
            if _BULLET_ITEM.match(following) or _ORDERED_ITEM.match(following) or following[:1] in (" ", "\t"):
                i += 1
                continue
            break
        if _ATX_HEADING.match(line) or _FENCE.match(line) or _looks_like_table_start(lines, i):
            break
        i += 1
    text = "\n".join(lines[start:i])
    return Block("list", text, start + 1, i, None), i


def _read_paragraph(lines, start):
    n = len(lines)
    i = start
    while i < n:
        line = lines[i]
        if line.strip() == "":
            break
        if i > start and (
            _ATX_HEADING.match(line)
            or _FENCE.match(line)
            or _BULLET_ITEM.match(line)
            or _ORDERED_ITEM.match(line)
            or _looks_like_table_start(lines, i)
        ):
            break
        i += 1
    text = "\n".join(lines[start:i])
    return Block("paragraph", text, start + 1, i, None), i
