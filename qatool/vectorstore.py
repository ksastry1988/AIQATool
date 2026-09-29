"""
Persistent local vector store wrapper.

The implementation keeps the rest of the codebase isolated from the storage
details while persisting indexed chunks and file hashes under a repo-local
`.qatool/index/store.json` file.
"""

from __future__ import annotations

import json
import math
import re
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
        tmp_path: Path | None = None
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
        try:
            tmp_path.replace(self._store_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

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
        with self._locked():
            indexed_files = set(self._data["file_hashes"])
            indexed_files.update(
                chunk.get("metadata", {}).get("file")
                for chunk in self._data["chunks"]
                if chunk.get("metadata", {}).get("file")
            )
            return indexed_files

    def get_file_hash(self, rel_path: str) -> str | None:
        """Return the last-indexed content hash for a file, if any."""
        with self._locked():
            return self._data["file_hashes"].get(rel_path)

    def snapshot(self) -> dict[str, Any]:
        with self._locked():
            indexed_files = set(self._data["file_hashes"])
            indexed_files.update(
                chunk.get("metadata", {}).get("file")
                for chunk in self._data["chunks"]
                if chunk.get("metadata", {}).get("file")
            )
            return {
                "file_hashes": dict(self._data["file_hashes"]),
                "indexed_files": indexed_files,
            }

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

    @staticmethod
    def _similarity(vector: list[float], chunk: dict[str, Any]) -> float:
        chunk_vector = chunk.get("vector", [])
        if len(chunk_vector) != len(vector):
            raise ValueError("query vector dimension mismatch")
        query_norm = math.sqrt(sum(value * value for value in vector))
        chunk_norm = math.sqrt(sum(value * value for value in chunk_vector))
        if query_norm == 0.0 or chunk_norm == 0.0:
            return 0.0
        return sum(a * b for a, b in zip(vector, chunk_vector)) / (query_norm * chunk_norm)

    @staticmethod
    def _result(chunk: dict[str, Any], score: float, match: str) -> dict[str, Any]:
        return {
            "id": chunk.get("id"),
            "text": chunk.get("text", ""),
            "metadata": dict(chunk.get("metadata", {})),
            "score": score,
            "match": match,
        }

    def query(self, vector: list[float], top_k: int = 8) -> list[Any]:
        """Return the top_k most similar chunks to the query vector."""
        if top_k <= 0:
            return []

        with self._locked():
            scored = [
                self._result(chunk, self._similarity(vector, chunk), "vector")
                for chunk in self._data["chunks"]
            ]

        scored.sort(key=lambda chunk: chunk["score"], reverse=True)
        return scored[:top_k]

    def find_exact(
        self,
        symbols: set[str],
        filenames: set[str],
        vector: list[float] | None = None,
    ) -> list[Any]:
        """Return chunks that exactly match a symbol or filename.

        A symbol matches the chunk's `symbol` metadata (case-sensitive), or,
        failing that, a definition of it in the chunk text (`def name`,
        `class Name`, `func name`, ...) so whole-file chunks without symbol
        metadata are still found. A filename matches the chunk's `file` path
        in full, as a trailing path suffix (`qatool/cli.py`), or by basename
        (`cli.py`). Ranking is symbol, then definition, then filename; within
        each group chunks are ordered by similarity to `vector` when given,
        else by file and line.
        """
        if not symbols and not filenames:
            return []

        definition = _definition_pattern(symbols)
        with self._locked():
            matches = []
            for chunk in self._data["chunks"]:
                metadata = chunk.get("metadata", {})
                if metadata.get("symbol") in symbols:
                    match = "symbol"
                elif definition is not None and definition.search(chunk.get("text", "")):
                    match = "definition"
                elif _file_matches(metadata.get("file"), filenames):
                    match = "filename"
                else:
                    continue
                score = self._similarity(vector, chunk) if vector is not None else 0.0
                matches.append(self._result(chunk, score, match))

        matches.sort(
            key=lambda result: (
                _MATCH_RANK[result["match"]],
                -result["score"],
                str(result["metadata"].get("file", "")),
                result["metadata"].get("start_line") or 0,
            )
        )
        return matches


_MATCH_RANK = {"symbol": 0, "definition": 1, "filename": 2}
_DEFINITION_KEYWORDS = (
    "def|class|function|func|fn|struct|enum|trait|interface|type|const|let|var"
)


def _definition_pattern(symbols: set[str]) -> re.Pattern[str] | None:
    """Match a keyword-introduced definition of any symbol (Go receivers allowed)."""
    if not symbols:
        return None
    names = "|".join(re.escape(symbol) for symbol in sorted(symbols))
    return re.compile(rf"\b(?:{_DEFINITION_KEYWORDS})\s+(?:\([^)]*\)\s*)?(?:{names})\b")


def _file_matches(rel_path: Any, filenames: set[str]) -> bool:
    if not isinstance(rel_path, str) or not filenames:
        return False
    for name in filenames:
        if rel_path == name or rel_path.endswith(f"/{name}"):
            return True
    return False
