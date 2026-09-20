"""
Thin wrapper around a vector database (Chroma, LanceDB, Qdrant, etc.)
so the rest of the codebase doesn't depend on any one library's API.

TODO: implement against your chosen backend. Chroma or LanceDB are the
simplest for a local/single-repo setup.
"""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any


class VectorStore:
    STORE_FILENAME = "store.json"

    def __init__(self, path: Path):
        self.path = path
        self._store_path = path / self.STORE_FILENAME
        self._data: dict[str, Any] = {
            "file_hashes": {},
            "chunks": [],
        }
        self._load()

    @classmethod
    def open(cls, path: Path) -> "VectorStore":
        path.mkdir(parents=True, exist_ok=True)
        return cls(path)

    def _load(self) -> None:
        if not self._store_path.exists():
            return

        raw = json.loads(self._store_path.read_text())
        self._data["file_hashes"] = dict(raw.get("file_hashes", {}))
        self._data["chunks"] = list(raw.get("chunks", []))

    def _persist(self) -> None:
        tmp_path = self._store_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self._data, indent=2, sort_keys=True))
        tmp_path.replace(self._store_path)

    def list_indexed_files(self) -> set[str]:
        indexed_files = set(self._data["file_hashes"])
        indexed_files.update(
            chunk.get("metadata", {}).get("file")
            for chunk in self._data["chunks"]
            if chunk.get("metadata", {}).get("file")
        )
        return indexed_files

    def get_file_hash(self, rel_path: str) -> str | None:
        """Return the last-indexed content hash for a file, if any."""
        return self._data["file_hashes"].get(rel_path)

    def set_file_hash(self, rel_path: str, new_hash: str) -> None:
        """Record the content hash used for a file's current chunks."""
        self._data["file_hashes"][rel_path] = new_hash
        self._persist()

    def delete_by_file(self, rel_path: str) -> None:
        """Remove all chunks previously indexed for this file."""
        chunks = [
            chunk
            for chunk in self._data["chunks"]
            if chunk.get("metadata", {}).get("file") != rel_path
        ]
        self._data["chunks"] = chunks
        self._data["file_hashes"].pop(rel_path, None)
        self._persist()

    def upsert(self, chunks: list[Any], vectors: list[list[float]]) -> None:
        """Insert or update chunks with their embedding vectors."""
        if len(chunks) != len(vectors):
            raise ValueError("chunk/vector count mismatch")

        for chunk, vector in zip(chunks, vectors):
            self._data["chunks"].append(
                {
                    "id": str(uuid.uuid4()),
                    "text": chunk.text,
                    "metadata": dict(chunk.metadata),
                    "vector": vector,
                }
            )
        self._persist()

    def query(self, vector: list[float], top_k: int = 8) -> list[Any]:
        """Return the top_k most similar chunks to the query vector."""
        if top_k <= 0:
            return []

        query_norm = math.sqrt(sum(value * value for value in vector))
        scored: list[dict[str, Any]] = []

        for chunk in self._data["chunks"]:
            chunk_vector = chunk.get("vector", [])
            if len(chunk_vector) != len(vector):
                raise ValueError("query vector dimension mismatch")
            chunk_norm = math.sqrt(sum(value * value for value in chunk_vector))
            if query_norm == 0.0 or chunk_norm == 0.0:
                score = 0.0
            else:
                score = sum(a * b for a, b in zip(vector, chunk_vector)) / (
                    query_norm * chunk_norm
                )

            scored.append(
                {
                    "text": chunk.get("text", ""),
                    "metadata": dict(chunk.get("metadata", {})),
                    "score": score,
                }
            )

        scored.sort(key=lambda chunk: chunk["score"], reverse=True)
        return scored[:top_k]
