import json

import pytest

from rag_chunker.chunker import _tail_overlap, chunk_markdown, chunks_to_jsonl
from rag_chunker.sentences import split_sentences


def test_max_tokens_must_be_positive():
    with pytest.raises(ValueError):
        chunk_markdown("text", max_tokens=0)
    with pytest.raises(ValueError):
        chunk_markdown("text", max_tokens=-5)


def test_overlap_must_be_non_negative_and_smaller_than_max_tokens():
    with pytest.raises(ValueError):
        chunk_markdown("text", overlap=-1)
    with pytest.raises(ValueError):
        chunk_markdown("text", max_tokens=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_markdown("text", max_tokens=10, overlap=20)


def test_empty_text_returns_no_chunks():
    assert chunk_markdown("") == []


def test_heading_prefix_is_prepended_to_text_but_not_body():
    text = "# Title\n\nHello world.\n"

    with_prefix = chunk_markdown(text)
    assert len(with_prefix) == 1
    assert with_prefix[0].body == "Hello world."
    assert with_prefix[0].text == "Title\n\nHello world."
    assert with_prefix[0].heading_path == ["Title"]
    assert not with_prefix[0].oversized

    without_prefix = chunk_markdown(text, heading_prefix=False)
    assert without_prefix[0].text == without_prefix[0].body == "Hello world."
    assert without_prefix[0].heading_path == ["Title"]


def test_chunk_never_crosses_a_heading_and_overlap_resets_there():
    text = "# A\n\nShort para one.\n\n## B\n\nShort para two.\n"
    chunks = chunk_markdown(text)

    assert len(chunks) == 2
    assert chunks[0].heading_path == ["A"]
    assert chunks[0].body == "Short para one."
    assert chunks[1].heading_path == ["A", "B"]
    assert chunks[1].text == "A > B\n\nShort para two."
    # the trailing prose of the first section is not carried into the next
    # heading's chunk, even though overlap is on by default
    assert chunks[1].body == "Short para two."


def test_code_table_and_oversized_paragraph_are_atomic():
    text = (
        "Intro sentence.\n"
        "\n"
        "```\n"
        "print('x')\n"
        "```\n"
        "\n"
        "| a | b |\n"
        "| - | - |\n"
        "| 1 | 2 |\n"
        "\n"
        "Outro sentence.\n"
    )
    # a one-token budget is smaller than any real block here, so every
    # block is forced to stand alone and is flagged oversized
    chunks = chunk_markdown(text, max_tokens=1, overlap=0, heading_prefix=False)

    assert [c.oversized for c in chunks] == [True, True, True, True]
    assert chunks[0].body == "Intro sentence."
    assert chunks[1].body == "```\nprint('x')\n```"
    assert chunks[2].body == "| a | b |\n| - | - |\n| 1 | 2 |"
    assert chunks[3].body == "Outro sentence."


def test_code_and_table_are_not_split_when_they_fit():
    text = "```\nprint('x')\n```\n\n| a | b |\n| - | - |\n| 1 | 2 |\n"
    chunks = chunk_markdown(text, heading_prefix=False)

    assert not any(c.oversized for c in chunks)
    bodies = "\n\n".join(c.body for c in chunks)
    assert "```\nprint('x')\n```" in bodies
    assert "| a | b |\n| - | - |\n| 1 | 2 |" in bodies


def test_long_paragraph_splits_on_sentence_boundaries_without_loss():
    sentence = "Alpha beta gamma delta."
    paragraph = " ".join([sentence] * 50)
    # the whole paragraph is far past the budget but every single sentence
    # is comfortably under it, so the fallback must split, never drop text
    chunks = chunk_markdown(paragraph + "\n", max_tokens=100, overlap=0, heading_prefix=False)

    assert len(chunks) > 1
    assert all(not c.oversized for c in chunks)
    assert all(c.token_estimate <= 100 for c in chunks)

    rebuilt = []
    for chunk in chunks:
        rebuilt.extend(split_sentences(chunk.body))
    assert rebuilt == split_sentences(paragraph)


def test_tail_overlap_pulls_trailing_sentences_up_to_the_budget():
    text = "One. Two. Three. Four."

    assert _tail_overlap(text, 0) == ""
    assert _tail_overlap(text, 1) == "Four."
    assert _tail_overlap(text, 100) == "One. Two. Three. Four."


def test_chunks_to_jsonl_serialises_index_and_fields_but_not_body():
    chunks = chunk_markdown("# T\n\nHello.\n", heading_prefix=False)
    output = chunks_to_jsonl(chunks)
    lines = output.split("\n")

    assert len(lines) == len(chunks) == 1
    record = json.loads(lines[0])
    assert record["index"] == 0
    assert record["text"] == chunks[0].text == "Hello."
    assert record["heading_path"] == ["T"]
    assert record["start_line"] == 3
    assert record["end_line"] == 3
    assert isinstance(record["token_estimate"], int) and record["token_estimate"] > 0
    assert "body" not in record
    assert "oversized" not in record
