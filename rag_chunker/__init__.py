"""Heading-aware markdown chunking for retrieval pipelines."""

from .chunker import Chunk, chunk_markdown, chunks_to_jsonl, chunks_to_text
from .markdown import Block, parse_blocks
from .sentences import split_sentences
from .tokens import estimate_tokens, fits_budget

__version__ = "0.2.0"

__all__ = [
    "Block",
    "Chunk",
    "__version__",
    "chunk_markdown",
    "chunks_to_jsonl",
    "chunks_to_text",
    "estimate_tokens",
    "fits_budget",
    "parse_blocks",
    "split_sentences",
]