"""Pack parsed markdown blocks into bounded, embeddable chunks.

A chunk never crosses a heading boundary, and it carries the heading path
that produced it so a downstream retriever knows what section a result
came from. A code block or table that alone exceeds the budget is emitted
whole and flagged ``oversized`` -- splitting a table off its header row
makes both halves useless, so there is no good way to cut it. A paragraph
in the same spot falls back to sentence boundaries instead, since prose
can be cut without losing its meaning. ``overlap`` repeats a slice of
trailing prose at the start of the next chunk so a search hit that lands
just past a chunk boundary still has some lead-in context, and it resets
at every heading so a sentence never carries into the wrong section.
"""

import json
from dataclasses import dataclass

from .markdown import parse_blocks
from .sentences import split_sentences
from .tokens import estimate_tokens

__all__ = ["Chunk", "chunk_markdown", "chunks_to_jsonl", "chunks_to_text"]


@dataclass(frozen=True)
class Chunk:
    """One packed unit of markdown, ready to embed.

    ``text`` is the heading path prefix (when enabled) plus ``body``;
    ``body`` alone is the content with no prefix. ``heading_path`` is the
    stack of enclosing heading titles, outermost first.
    """

    text: str
    body: str
    heading_path: list
    start_line: int
    end_line: int
    token_estimate: int
    oversized: bool

    def to_dict(self):
        return {
            "text": self.text,
            "heading_path": self.heading_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "token_estimate": self.token_estimate,
        }


def chunk_markdown(text, max_tokens=512, overlap=64, heading_prefix=True):
    """Split ``text`` into a list of :class:`Chunk`.

    Chunks never cross a heading. A code block, table or list that alone
    exceeds ``max_tokens`` is emitted whole with ``oversized=True``; an
    oversized paragraph falls back to sentence boundaries instead.
    ``overlap`` is the number of trailing tokens of a chunk's prose that
    are repeated at the start of the next chunk in the same section.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")

    chunks = []
    heading_stack = []  # list of (level, title)
    pieces = []  # body fragments accumulated for the chunk in progress
    pieces_prose = []  # parallel to pieces: True where the fragment is prose
    start_line = end_line = None
    pending_overlap = ""

    def current_heading_path():
        return [title for _level, title in heading_stack]

    def render(body):
        if heading_prefix and heading_stack:
            prefix = " > ".join(current_heading_path())
            return f"{prefix}\n\n{body}" if body else prefix
        return body

    def fits(candidate_pieces):
        return estimate_tokens(render("\n\n".join(candidate_pieces))) <= max_tokens

    def flush():
        nonlocal pieces, pieces_prose, start_line, end_line, pending_overlap
        if not pieces:
            return
        body = "\n\n".join(pieces)
        full_text = render(body)
        chunks.append(
            Chunk(
                text=full_text,
                body=body,
                heading_path=current_heading_path(),
                start_line=start_line,
                end_line=end_line,
                token_estimate=estimate_tokens(full_text),
                oversized=False,
            )
        )
        pending_overlap = _tail_overlap(pieces[-1], overlap) if pieces_prose[-1] else ""
        pieces = []
        pieces_prose = []
        start_line = end_line = None

    def emit_oversized(atom_text, atom_start, atom_end):
        nonlocal pending_overlap
        flush()
        full_text = render(atom_text)
        chunks.append(
            Chunk(
                text=full_text,
                body=atom_text,
                heading_path=current_heading_path(),
                start_line=atom_start,
                end_line=atom_end,
                token_estimate=estimate_tokens(full_text),
                oversized=True,
            )
        )
        # Repeating a table or code block into the next chunk would just
        # duplicate it wholesale, so an oversized atom carries no overlap.
        pending_overlap = ""

    def add_atom(atom_text, atom_start, atom_end, prose, oversized_alone):
        nonlocal pieces, pieces_prose, start_line, end_line, pending_overlap
        if oversized_alone:
            emit_oversized(atom_text, atom_start, atom_end)
            return
        if pieces and fits(pieces + [atom_text]):
            pieces.append(atom_text)
            pieces_prose.append(prose)
            end_line = atom_end
            return
        if pieces:
            flush()
        candidate = ([pending_overlap] if pending_overlap else []) + [atom_text]
        if len(candidate) > 1 and not fits(candidate):
            candidate = [atom_text]
        pieces = candidate
        pieces_prose = [True] * (len(candidate) - 1) + [prose]
        start_line = atom_start
        end_line = atom_end
        pending_overlap = ""

    for block in parse_blocks(text):
        if block.kind == "heading":
            flush()
            level = block.level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, block.text))
            # A heading always starts a clean section; nothing carries over.
            pending_overlap = ""
            continue

        if block.kind == "paragraph":
            if estimate_tokens(render(block.text)) <= max_tokens:
                add_atom(block.text, block.start_line, block.end_line, True, False)
            else:
                for group_text, group_start, group_end, group_oversized in _split_paragraph(
                    block, max_tokens, render
                ):
                    add_atom(group_text, group_start, group_end, True, group_oversized)
            continue

        # code, table and list blocks are atomic: never split, only flagged.
        solo_tokens = estimate_tokens(render(block.text))
        add_atom(block.text, block.start_line, block.end_line, False, solo_tokens > max_tokens)

    flush()
    return chunks


def _split_paragraph(block, max_tokens, render):
    """Break a paragraph too big for one chunk into sentence groups.

    Each group is packed to fit under ``max_tokens``. A single sentence
    that alone still does not fit is kept whole and marked oversized,
    since cutting mid-sentence would defeat the point of the fallback.
    """
    sentences = split_sentences(block.text)
    if not sentences:
        return [(block.text, block.start_line, block.end_line, True)]
    groups = []
    current = []
    for sentence in sentences:
        candidate = current + [sentence]
        if current and estimate_tokens(render(" ".join(candidate))) > max_tokens:
            groups.append(current)
            current = [sentence]
        else:
            current = candidate
    groups.append(current)
    return [
        (
            " ".join(group),
            block.start_line,
            block.end_line,
            estimate_tokens(render(" ".join(group))) > max_tokens,
        )
        for group in groups
    ]


def _tail_overlap(text, overlap_tokens):
    """Return trailing sentences of ``text`` totalling roughly ``overlap_tokens``."""
    if overlap_tokens <= 0:
        return ""
    sentences = split_sentences(text)
    if not sentences:
        return ""
    tail = []
    total = 0
    for sentence in reversed(sentences):
        tail.insert(0, sentence)
        total += estimate_tokens(sentence)
        if total >= overlap_tokens:
            break
    return " ".join(tail)


def _records(chunks):
    """Return each chunk as a dict with its ``index`` prepended."""
    records = []
    for index, chunk in enumerate(chunks):
        record = {"index": index}
        record.update(chunk.to_dict())
        records.append(record)
    return records


def chunks_to_jsonl(chunks):
    """Serialise ``chunks`` as newline-delimited JSON, one object per line.

    Each line is the chunk's own ``to_dict()`` fields plus its position in
    the list as ``index``, since a chunk on its own does not know where it
    sits among its siblings.
    """
    return "\n".join(json.dumps(record) for record in _records(chunks))


def chunks_to_text(chunks, separator="\n\n"):
    """Return the rendered ``text`` of each chunk, joined by ``separator``.

    This is the plain-text form of the same output the JSON formats carry,
    for pipelines that want to drop chunks straight into an embedder or a
    vector store without parsing JSONL. Each chunk keeps its heading path
    prefix (unless ``heading_prefix`` was disabled) and is separated from
    its neighbours by ``separator`` (two newlines by default).
    """
    return separator.join(chunk.text for chunk in chunks)
