"""
Thin wrapper around a vector database (Chroma, LanceDB, Qdrant, etc.)
so the rest of the codebase doesn't depend on any one library's API.

TODO: implement against your chosen backend. Chroma or LanceDB are the
simplest for a local/single-repo setup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class VectorStore:
    def __init__(self, path: Path):
        self.path = path
        # TODO: open/create the underlying DB at `path`

    @classmethod
    def open(cls, path: Path) -> "VectorStore":
        path.mkdir(parents=True, exist_ok=True)
        return cls(path)

    def get_file_hash(self, rel_path: str) -> str | None:
        """Return the last-indexed content hash for a file, if any."""
        raise NotImplementedError

    def set_file_hash(self, rel_path: str, new_hash: str) -> None:
        """Record the content hash used for a file's current chunks."""
        raise NotImplementedError

    def delete_by_file(self, rel_path: str) -> None:
        """Remove all chunks previously indexed for this file."""
        raise NotImplementedError

    def upsert(self, chunks: list[Any], vectors: list[list[float]]) -> None:
        """Insert or update chunks with their embedding vectors."""
        raise NotImplementedError

    def query(self, vector: list[float], top_k: int = 8) -> list[Any]:
        """Return the top_k most similar chunks to the query vector."""
        raise NotImplementedError
