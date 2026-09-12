import json

import pytest

from rag_chunker import cli


def _write(tmp_path, text, name="doc.md"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_default_output_is_jsonl_on_stdout(tmp_path, capsys):
    path = _write(tmp_path, "# Title\n\nHello world.\n")

    exit_code = cli.main([path])

    assert exit_code == 0
    out = capsys.readouterr().out
    lines = out.strip("\n").split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["index"] == 0
    assert record["heading_path"] == ["Title"]


def test_array_flag_emits_one_indented_json_array(tmp_path, capsys):
    path = _write(tmp_path, "# Title\n\nHello world.\n")

    cli.main([path, "--array"])

    out = capsys.readouterr().out
    records = json.loads(out)
    assert isinstance(records, list)
    assert records[0]["heading_path"] == ["Title"]


def test_no_heading_prefix_flag_is_passed_through(tmp_path, capsys):
    path = _write(tmp_path, "# Title\n\nHello world.\n")

    cli.main([path, "--no-heading-prefix", "--array"])

    records = json.loads(capsys.readouterr().out)
    assert records[0]["text"] == "Hello world."


class _FakeStdin:
    def __init__(self, text):
        self._text = text

    def read(self):
        return self._text


def test_reads_stdin_when_path_is_dash(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin("# T\n\nBody.\n"))

    cli.main(["-"])

    out = capsys.readouterr().out
    record = json.loads(out.strip())
    assert record["heading_path"] == ["T"]


def test_output_option_writes_to_file_instead_of_stdout(tmp_path, capsys):
    src = _write(tmp_path, "# T\n\nBody.\n")
    dest = tmp_path / "out.jsonl"

    cli.main([src, "-o", str(dest)])

    assert capsys.readouterr().out == ""
    record = json.loads(dest.read_text(encoding="utf-8").strip())
    assert record["heading_path"] == ["T"]


def test_stats_flag_prints_summary_to_stderr(tmp_path, capsys):
    path = _write(tmp_path, "# T\n\nBody.\n")

    cli.main([path, "--stats"])

    err = capsys.readouterr().err.strip()
    assert err.endswith("oversized")
    assert "1 chunks" in err


def test_missing_file_exits_with_error(tmp_path, capsys):
    missing = str(tmp_path / "does-not-exist.md")

    with pytest.raises(SystemExit) as excinfo:
        cli.main([missing])

    assert excinfo.value.code != 0
    assert "cannot read" in capsys.readouterr().err


def test_invalid_max_tokens_exits_with_error(tmp_path, capsys):
    path = _write(tmp_path, "Body.\n")

    with pytest.raises(SystemExit) as excinfo:
        cli.main([path, "--max-tokens", "0"])

    assert excinfo.value.code != 0
    assert "max_tokens must be a positive integer" in capsys.readouterr().err


def test_format_text_emits_chunk_text_joined_by_blank_lines(tmp_path, capsys):
    path = _write(tmp_path, "# Title\n\nHello world.\n\n## Sub\n\nBody line.\n")

    cli.main([path, "--format", "text"])

    out = capsys.readouterr().out
    assert out == "Title\n\nHello world.\n\nTitle > Sub\n\nBody line.\n"


def test_format_array_matches_array_flag(tmp_path, capsys):
    path = _write(tmp_path, "# T\n\nBody.\n")

    cli.main([path, "--format", "array"])
    via_format = json.loads(capsys.readouterr().out)

    cli.main([path, "--array"])
    via_flag = json.loads(capsys.readouterr().out)

    assert via_format == via_flag


def test_stats_line_reports_zero_chunks_for_empty_input():
    assert cli._stats_line([]) == "0 chunks"
