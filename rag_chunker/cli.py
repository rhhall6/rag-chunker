"""Command-line entry point: ``rag-chunker`` / ``python -m rag_chunker.cli``.

Reads a markdown document (a file path or ``-`` for stdin), chunks it, and
writes the result as JSON -- one object per line by default, matching the
JSONL convention most embedding pipelines already read.
"""

import argparse
import json
import sys

from .chunker import chunk_markdown, chunks_to_jsonl


def build_parser():
    parser = argparse.ArgumentParser(
        prog="rag-chunker",
        description="Split a markdown document into bounded, embeddable chunks.",
    )
    parser.add_argument("path", help="markdown file to chunk, or - to read stdin")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="chunk size ceiling, heading prefix included (default: 512)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="trailing tokens repeated in the next chunk of a section (default: 64)",
    )
    parser.add_argument(
        "--no-heading-prefix",
        action="store_true",
        help="do not prepend the heading path to the chunk text",
    )
    parser.add_argument(
        "--array",
        action="store_true",
        help="emit one indented JSON array instead of JSON lines",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="print a size summary to stderr",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="write the result to a file instead of stdout",
    )
    return parser


def _read_input(path):
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _render(chunks, as_array):
    if not as_array:
        return chunks_to_jsonl(chunks)
    records = []
    for index, chunk in enumerate(chunks):
        record = {"index": index}
        record.update(chunk.to_dict())
        records.append(record)
    return json.dumps(records, indent=2)


def _stats_line(chunks):
    if not chunks:
        return "0 chunks"
    tokens = [chunk.token_estimate for chunk in chunks]
    oversized = sum(1 for chunk in chunks if chunk.oversized)
    avg = round(sum(tokens) / len(tokens))
    return (
        f"{len(chunks)} chunks | tokens min {min(tokens)} avg {avg} max {max(tokens)} "
        f"| {oversized} oversized"
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        text = _read_input(args.path)
    except OSError as exc:
        parser.error(f"cannot read {args.path}: {exc}")

    try:
        chunks = chunk_markdown(
            text,
            max_tokens=args.max_tokens,
            overlap=args.overlap,
            heading_prefix=not args.no_heading_prefix,
        )
    except ValueError as exc:
        parser.error(str(exc))

    output = _render(chunks, args.array)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
            handle.write("\n")
    else:
        print(output)

    if args.stats:
        print(_stats_line(chunks), file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
