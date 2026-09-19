"""
Language-aware chunking.

Splits a source file into semantically coherent chunks (functions,
classes, top-level blocks) rather than fixed-size windows, so retrieval
returns whole units of meaning instead of arbitrary line slices.

TODO: implement using tree-sitter grammars per language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_file(full_path: Path, rel_path: str) -> list[Chunk]:
    """
    Return a list of Chunks for one file.

    Each chunk's metadata should include at minimum:
        - file: rel_path
        - start_line / end_line
        - symbol: function/class name, if applicable
        - language
    """
    # Placeholder: treat the whole file as one chunk until tree-sitter
    # parsing is wired in.
    text = full_path.read_text(errors="ignore")
    return [
        Chunk(
            text=text,
            metadata={
                "file": rel_path,
                "start_line": 1,
                "end_line": text.count("\n") + 1,
                "symbol": None,
                "language": full_path.suffix.lstrip("."),
            },
        )
    ]
