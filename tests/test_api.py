"""Public API surface: everything documented as a top-level export must be
importable from ``rag_chunker`` (not just its submodule), and the plain-text
output format must behave."""

import rag_chunker
from rag_chunker import (
    Block,
    Chunk,
    chunk_markdown,
    chunks_to_jsonl,
    chunks_to_text,
    estimate_tokens,
    fits_budget,
    parse_blocks,
    split_sentences,
)


def test_documented_top_level_exports_exist():
    # README's "Library API" section lists these as available straight off
    # the package, so each must be importable from the root namespace.
    for name in [
        "Block",
        "Chunk",
        "chunk_markdown",
        "chunks_to_jsonl",
        "chunks_to_text",
        "estimate_tokens",
        "fits_budget",
        "parse_blocks",
        "split_sentences",
        "__version__",
    ]:
        assert name in rag_chunker.__all__
        assert hasattr(rag_chunker, name), f"{name} missing from package root"


def test_public_api_types_are_shared_across_modules():
    # The exports must be the very instances the submodules define, not
    # lookalikes -- otherwise isinstance checks and pickling break.
    assert rag_chunker.Chunk is rag_chunker.chunker.Chunk
    assert rag_chunker.parse_blocks is rag_chunker.markdown.parse_blocks
    assert rag_chunker.chunk_markdown is rag_chunker.chunker.chunk_markdown


def test_chunks_to_text_joins_rendered_text_with_default_separator():
    text = "# Title\n\nHello world.\n\n## Sub\n\nBody line.\n"
    chunks = chunk_markdown(text)

    assert chunks_to_text(chunks) == "Title\n\nHello world.\n\nTitle > Sub\n\nBody line."


def test_chunks_to_text_heading_prefix_off():
    text = "# Title\n\nHello world.\n"
    chunks = chunk_markdown(text, heading_prefix=False)

    assert chunks_to_text(chunks) == "Hello world."


def test_chunks_to_text_custom_separator():
    chunks = chunk_markdown("# A\n\nOne.\n\n## B\n\nTwo.\n")

    # separator.join semantics: the separator is inserted verbatim between
    # chunk texts (each of which may itself contain blank lines).
    assert chunks_to_text(chunks, separator="\n---\n") == (
        "A\n\nOne.\n---\nA > B\n\nTwo."
    )


def test_chunks_to_text_empty_input():
    assert chunks_to_text([]) == ""


def test_jsonl_and_text_carry_the_same_text_field():
    # A pipeline can switch from JSON to plain text without changing what
    # gets embedded: the ``text`` field in JSONL equals the text output.
    import json

    doc = "# A\n\nOne.\n\n## B\n\nTwo.\n"
    chunks = chunk_markdown(doc)

    # a sentinel separator lets us split the text output back into chunks
    sentinel = "\x00\x00"
    jsonl_texts = [json.loads(line)["text"] for line in chunks_to_jsonl(chunks).splitlines()]
    assert jsonl_texts == chunks_to_text(chunks, separator=sentinel).split(sentinel)