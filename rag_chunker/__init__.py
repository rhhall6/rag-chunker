"""Heading-aware markdown chunking for retrieval pipelines."""

# chunk_markdown and friends land in .chunker, once it exists; that module
# ties this parser together with the token estimator and sentence splitter.
from .markdown import Block, parse_blocks
from .sentences import split_sentences
from .tokens import estimate_tokens, fits_budget

__version__ = "0.1.0"

__all__ = [
    "Block",
    "__version__",
    "estimate_tokens",
    "fits_budget",
    "parse_blocks",
    "split_sentences",
]
