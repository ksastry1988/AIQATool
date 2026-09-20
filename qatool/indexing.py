from __future__ import annotations

import fnmatch
import hashlib
import sys
from pathlib import Path

from .chunking import Chunk, chunk_file
from .embeddings import embed_texts
from .vectorstore import VectorStore

IGNORED_SUFFIXES = {".lock", ".png", ".jpg", ".svg", ".pyc"}
IGNORED_NAME_SUFFIXES = (".min.js", ".map")
IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".qatool",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
}
IGNORE_FILE = ".qatoolignore"
INDEX_FILE_BATCH_SIZE = 32
SECRET_FILENAMES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".npmrc",
    "id_rsa",
    "id_dsa",
    "id_ed25519",
}
SECRET_SUFFIXES = {
    ".key",
    ".pem",
    ".p12",
    ".pfx",
    ".crt",
    ".cer",
    ".der",
    ".jks",
    ".keystore",
}


def load_ignore_patterns(repo_root: Path) -> set[str]:
    ignore_path = repo_root / IGNORE_FILE
    if not ignore_path.exists():
        return set()
    return {
        line.strip()
        for line in ignore_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def _matches_ignore_pattern(rel_path: str, pattern: str) -> bool:
    normalized = pattern.strip().lstrip("./")
    if not normalized:
        return False

    if normalized.endswith("/"):
        prefix = normalized.rstrip("/")
        return rel_path == prefix or rel_path.startswith(f"{prefix}/")

    if fnmatch.fnmatch(rel_path, normalized):
        return True

    if "/" not in normalized and not any(char in normalized for char in "*?[]"):
        parts = rel_path.split("/")
        return normalized in parts or Path(rel_path).name == normalized

    return False


def _is_binary_file(path: Path) -> bool:
    sample = path.read_bytes()[:8192]
    if not sample:
        return False
    if b"\x00" in sample:
        return True

    text_bytes = sum(
        byte in {9, 10, 13} or 32 <= byte <= 126
        for byte in sample
    )
    return (text_bytes / len(sample)) < 0.75


def should_index(path: Path, repo_root: Path, ignore_patterns: set[str]) -> bool:
    rel_path = path.relative_to(repo_root).as_posix()

    if rel_path == IGNORE_FILE:
        return False
    if any(part in IGNORED_DIRECTORIES for part in path.relative_to(repo_root).parts[:-1]):
        return False
    if path.suffix in IGNORED_SUFFIXES:
        return False
    if path.name.endswith(IGNORED_NAME_SUFFIXES):
        return False
    if path.name in SECRET_FILENAMES or path.suffix in SECRET_SUFFIXES:
        return False
    if any(_matches_ignore_pattern(rel_path, pattern) for pattern in ignore_patterns):
        return False
    return not _is_binary_file(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_repository_path(repo_root: Path) -> Path:
    if not repo_root.exists():
        raise ValueError(f"repository does not exist: {repo_root}")
    if not repo_root.is_dir():
        raise ValueError(f"repository path is not a directory: {repo_root}")
    return repo_root


def iter_candidate_files(repo_root: Path, ignore_patterns: set[str]) -> tuple[list[Path], list[str]]:
    candidates: list[Path] = []
    errors: list[str] = []

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue

        rel_path = path.relative_to(repo_root).as_posix()
        try:
            if should_index(path, repo_root, ignore_patterns):
                candidates.append(path)
        except OSError as exc:
            errors.append(f"{rel_path}: failed to inspect file ({exc})")

    return candidates, errors


def _embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    vectors = embed_texts([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError(
            f"embedding count mismatch: expected {len(chunks)}, got {len(vectors)}"
        )
    return vectors


def _index_file_batch(
    batch: list[tuple[str, Path, str]],
    store: VectorStore,
    errors: list[str],
    start_index: int,
    total_files: int,
) -> tuple[int, int, int]:
    prepared: list[tuple[str, str, list]] = []
    indexed_files = 0
    indexed_chunks = 0
    deleted_files = 0

    for offset, (rel_path, full_path, new_hash) in enumerate(batch):
        print(f"[qatool] indexing {start_index + offset + 1}/{total_files}: {rel_path}")
        try:
            chunks = chunk_file(full_path, rel_path)
        except FileNotFoundError:
            store.delete_by_file(rel_path)
            deleted_files += 1
        except Exception as exc:  # pragma: no cover - defensive boundary
            errors.append(f"{rel_path}: failed to index file ({exc})")
        else:
            prepared.append((rel_path, new_hash, chunks))

    if not prepared:
        return indexed_files, indexed_chunks, deleted_files

    try:
        flattened_chunks = [chunk for _, _, chunks in prepared for chunk in chunks]
        vectors = _embed_chunks(flattened_chunks)
    except Exception:
        for rel_path, new_hash, chunks in prepared:
            try:
                store.replace_file(rel_path, new_hash, chunks, _embed_chunks(chunks))
                indexed_files += 1
                indexed_chunks += len(chunks)
            except Exception as exc:  # pragma: no cover - defensive boundary
                errors.append(f"{rel_path}: failed to index file ({exc})")
        return indexed_files, indexed_chunks, deleted_files

    vector_offset = 0
    for rel_path, new_hash, chunks in prepared:
        next_offset = vector_offset + len(chunks)
        try:
            store.replace_file(rel_path, new_hash, chunks, vectors[vector_offset:next_offset])
            indexed_files += 1
            indexed_chunks += len(chunks)
        except Exception as exc:  # pragma: no cover - defensive boundary
            errors.append(f"{rel_path}: failed to index file ({exc})")
        vector_offset = next_offset

    return indexed_files, indexed_chunks, deleted_files


def index_repository(repo_root: Path, store: VectorStore) -> None:
    repo_root = validate_repository_path(repo_root)

    ignore_patterns = load_ignore_patterns(repo_root)
    candidates, errors = iter_candidate_files(repo_root, ignore_patterns)
    current_files = {
        path.relative_to(repo_root).as_posix(): path
        for path in sorted(candidates)
    }

    snapshot = store.snapshot()
    stale_files = sorted(snapshot["indexed_files"] - set(current_files))

    to_process: list[tuple[str, Path, str]] = []
    unchanged = 0

    for rel_path, full_path in current_files.items():
        try:
            new_hash = file_hash(full_path)
        except OSError as exc:
            errors.append(f"{rel_path}: failed to read file ({exc})")
            continue

        if snapshot["file_hashes"].get(rel_path) == new_hash:
            unchanged += 1
            continue

        to_process.append((rel_path, full_path, new_hash))

    print(
        f"[qatool] scanned {len(current_files)} file(s); "
        f"{len(to_process)} to index, {unchanged} unchanged, "
        f"{len(stale_files)} stale removal(s)"
    )

    indexed_files = 0
    indexed_chunks = 0
    deleted_files = 0
    for start in range(0, len(to_process), INDEX_FILE_BATCH_SIZE):
        batch = to_process[start:start + INDEX_FILE_BATCH_SIZE]
        batch_indexed_files, batch_indexed_chunks, batch_deleted_files = _index_file_batch(
            batch,
            store,
            errors,
            start,
            len(to_process),
        )
        indexed_files += batch_indexed_files
        indexed_chunks += batch_indexed_chunks
        deleted_files += batch_deleted_files

    for rel_path in stale_files:
        store.delete_by_file(rel_path)
        deleted_files += 1

    if errors:
        for error in errors:
            print(f"[qatool] {error}", file=sys.stderr)

    print(
        f"[qatool] indexed {indexed_files} file(s), "
        f"{indexed_chunks} chunk(s), {deleted_files} deletion(s), "
        f"{len(errors)} error(s)"
    )
