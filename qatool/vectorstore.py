"""
Thin wrapper around a vector database (Chroma, LanceDB, Qdrant, etc.)
so the rest of the codebase doesn't depend on any one library's API.

TODO: implement against your chosen backend. Chroma or LanceDB are the
simplest for a local/single-repo setup.
"""

from __future__ import annotations

import json
import math
import tempfile
import uuid
from contextlib import contextmanager
from io import BufferedRandom
from pathlib import Path
from typing import Any

try:  # pragma: no cover - import path depends on platform
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

try:  # pragma: no cover - import path depends on platform
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None


class VectorStore:
    STORE_FILENAME = "store.json"

    def __init__(self, path: Path):
        self.path = path
        self._store_path = path / self.STORE_FILENAME
        self._lock_path = path / f"{self.STORE_FILENAME}.lock"
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
        self._data = {
            "file_hashes": {},
            "chunks": [],
        }
        if not self._store_path.exists():
            return

        raw = json.loads(self._store_path.read_text())
        self._data["file_hashes"] = dict(raw.get("file_hashes", {}))
        self._data["chunks"] = list(raw.get("chunks", []))

    def _persist(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.path,
            prefix=f"{self._store_path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_file.write(json.dumps(self._data, indent=2, sort_keys=True))
            tmp_path = Path(tmp_file.name)
        tmp_path.replace(self._store_path)

    def _acquire_lock(self, lock_file: BufferedRandom) -> None:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            return

        if msvcrt is not None:
            if self._lock_path.stat().st_size == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            return

    def _release_lock(self, lock_file: BufferedRandom) -> None:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            return

        if msvcrt is not None:
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)

    @contextmanager
    def _locked(self):
        self.path.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+b") as lock_file:
            self._acquire_lock(lock_file)
            try:
                self._load()
                yield
            finally:
                self._release_lock(lock_file)

    def _chunk_record(self, chunk: Any, vector: list[float]) -> dict[str, Any]:
        return {
            "id": str(uuid.uuid4()),
            "text": chunk.text,
            "metadata": dict(chunk.metadata),
            "vector": vector,
        }

    def list_indexed_files(self) -> set[str]:
        self._load()
        indexed_files = set(self._data["file_hashes"])
        indexed_files.update(
            chunk.get("metadata", {}).get("file")
            for chunk in self._data["chunks"]
            if chunk.get("metadata", {}).get("file")
        )
        return indexed_files

    def get_file_hash(self, rel_path: str) -> str | None:
        """Return the last-indexed content hash for a file, if any."""
        self._load()
        return self._data["file_hashes"].get(rel_path)

    def set_file_hash(self, rel_path: str, new_hash: str) -> None:
        """Record the content hash used for a file's current chunks."""
        with self._locked():
            self._data["file_hashes"][rel_path] = new_hash
            self._persist()

    def delete_by_file(self, rel_path: str) -> None:
        """Remove all chunks previously indexed for this file."""
        with self._locked():
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

        updated_files = {
            chunk.metadata.get("file")
            for chunk in chunks
            if chunk.metadata.get("file")
        }

        with self._locked():
            self._data["chunks"] = [
                chunk
                for chunk in self._data["chunks"]
                if chunk.get("metadata", {}).get("file") not in updated_files
            ]

            for chunk, vector in zip(chunks, vectors):
                self._data["chunks"].append(self._chunk_record(chunk, vector))

            self._persist()

    def replace_file(
        self,
        rel_path: str,
        new_hash: str,
        chunks: list[Any],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunk/vector count mismatch")

        replacement_chunks = [self._chunk_record(chunk, vector) for chunk, vector in zip(chunks, vectors)]

        with self._locked():
            self._data["chunks"] = [
                chunk
                for chunk in self._data["chunks"]
                if chunk.get("metadata", {}).get("file") != rel_path
            ]
            self._data["chunks"].extend(replacement_chunks)
            self._data["file_hashes"][rel_path] = new_hash
            self._persist()

    def query(self, vector: list[float], top_k: int = 8) -> list[Any]:
        """Return the top_k most similar chunks to the query vector."""
        if top_k <= 0:
            return []

        self._load()
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
