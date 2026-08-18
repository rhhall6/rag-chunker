from rag_chunker.markdown import Block, parse_blocks


def test_empty_input():
    assert parse_blocks("") == []
    assert parse_blocks("\n\n\n") == []


def test_heading_levels_and_trailing_hashes():
    text = "# One\n\n## Two ##\n\n###### Six\n"
    blocks = parse_blocks(text)
    assert [b.kind for b in blocks] == ["heading", "heading", "heading"]
    assert [b.level for b in blocks] == [1, 2, 6]
    assert [b.text for b in blocks] == ["One", "Two", "Six"]


def test_heading_requires_a_space():
    # no space after the hashes -- not an ATX heading, falls through to paragraph
    blocks = parse_blocks("#nope\n")
    assert blocks == [Block("paragraph", "#nope", 1, 1, None)]


def test_empty_heading():
    blocks = parse_blocks("#\n")
    assert blocks == [Block("heading", "", 1, 1, 1)]


def test_paragraph_breaks_before_heading_without_blank_line():
    blocks = parse_blocks("Text line\n# Heading\n")
    assert blocks == [
        Block("paragraph", "Text line", 1, 1, None),
        Block("heading", "Heading", 2, 2, 1),
    ]


def test_setext_heading_is_read_as_a_paragraph():
    text = "Title\n=====\n\nBody text.\n"
    blocks = parse_blocks(text)
    assert blocks == [
        Block("paragraph", "Title\n=====", 1, 2, None),
        Block("paragraph", "Body text.", 4, 4, None),
    ]


def test_fenced_code_block_is_atomic():
    text = "```python\nprint(1)\n```\n"
    blocks = parse_blocks(text)
    assert blocks == [Block("code", "```python\nprint(1)\n```", 1, 3, None)]


def test_unterminated_fence_runs_to_eof():
    text = "```\nprint(1)\n"
    blocks = parse_blocks(text)
    assert blocks == [Block("code", "```\nprint(1)", 1, 2, None)]


def test_tilde_fence_closed_by_longer_fence():
    text = "~~~\ncode\n~~~~\n"
    blocks = parse_blocks(text)
    assert blocks == [Block("code", "~~~\ncode\n~~~~", 1, 3, None)]


def test_fence_marker_mismatch_does_not_close():
    # a tilde run inside a backtick fence is just content, not a close
    text = "```\n~~~\nstill code\n```\n"
    blocks = parse_blocks(text)
    assert blocks == [Block("code", "```\n~~~\nstill code\n```", 1, 4, None)]


def test_pipe_table_is_atomic():
    text = "| a | b |\n| - | - |\n| 1 | 2 |\n"
    blocks = parse_blocks(text)
    assert blocks == [Block("table", "| a | b |\n| - | - |\n| 1 | 2 |", 1, 3, None)]


def test_bullet_list_run_with_internal_blank_line():
    text = "- one\n- two\n\n- three\n"
    blocks = parse_blocks(text)
    assert blocks == [Block("list", "- one\n- two\n\n- three", 1, 4, None)]


def test_ordered_list_run():
    text = "1. one\n2. two\n"
    blocks = parse_blocks(text)
    assert blocks == [Block("list", "1. one\n2. two", 1, 2, None)]


def test_list_ends_at_heading():
    text = "- one\n- two\n# Next\n"
    blocks = parse_blocks(text)
    assert blocks[0] == Block("list", "- one\n- two", 1, 2, None)
    assert blocks[1] == Block("heading", "Next", 3, 3, 1)


def test_paragraph_lines_are_joined():
    text = "Line one\nLine two\nLine three\n"
    blocks = parse_blocks(text)
    assert blocks == [
        Block("paragraph", "Line one\nLine two\nLine three", 1, 3, None)
    ]


def test_full_document_order_and_line_numbers():
    text = (
        "# Runbook\n"
        "\n"
        "Intro paragraph.\n"
        "\n"
        "## Checks\n"
        "\n"
        "| name | status |\n"
        "| --- | --- |\n"
        "| disk | ok |\n"
        "\n"
        "- check one\n"
        "- check two\n"
    )
    blocks = parse_blocks(text)
    kinds = [b.kind for b in blocks]
    assert kinds == ["heading", "paragraph", "heading", "table", "list"]
    assert blocks[0].start_line == 1
    assert blocks[1] == Block("paragraph", "Intro paragraph.", 3, 3, None)
    assert blocks[2] == Block("heading", "Checks", 5, 5, 2)
    assert blocks[3].start_line == 7 and blocks[3].end_line == 9
    assert blocks[4].start_line == 11 and blocks[4].end_line == 12
