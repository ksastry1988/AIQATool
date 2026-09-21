"""
qatool.reindex

Incremental re-indexing for the Codebase Q&A tool.

Called by the git post-commit/post-merge hook with the list of files
that changed. Only re-chunks, re-embeds, and upserts those files
instead of rebuilding the whole vector index.

Usage (invoked by the hook):
    qatool reindex --repo /path/to/repo --changed-files /path/to/diff.txt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Swap these for your real implementations.
from .chunking import chunk_file          # tree-sitter based, function/class-level chunks
from .embeddings import embed_texts       # batched embedding calls
from .indexing import (
    file_hash,
    load_ignore_patterns,
    should_index,
    validate_repository_path,
)
from .vectorstore import VectorStore      # thin wrapper around Chroma/LanceDB/etc.


@dataclass
class FileChange:
    status: str   # 'A' added, 'M' modified, 'D' deleted, 'R' renamed
    path: str
    new_path: str | None = None  # set for renames


def parse_diff_output(diff_text: str) -> list[FileChange]:
    """Parse `git diff --name-status` output into FileChange objects."""
    changes = []
    for line in diff_text.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0][0]  # 'R100' -> 'R'
        if status == "R":
            changes.append(FileChange(status="R", path=parts[1], new_path=parts[2]))
        else:
            changes.append(FileChange(status=status, path=parts[1]))
    return changes


def reindex(repo_root: Path, changes: list[FileChange], store: VectorStore) -> None:
    ignore_patterns = load_ignore_patterns(repo_root)

    added_or_modified: dict[str, Path] = {}
    deleted: set[str] = set()

    for change in changes:
        if change.status == "D":
            deleted.add(change.path)
            continue

        target = change.new_path if change.status == "R" else change.path
        full_path = repo_root / target

        if change.status == "R" and change.path != change.new_path:
            # Old path's chunks are stale regardless of content.
            deleted.add(change.path)

        if not full_path.exists():
            continue  # file was deleted after the diff was captured
        if not should_index(full_path, repo_root, ignore_patterns):
            continue

        added_or_modified[target] = full_path

    # 1. Remove stale chunks for deleted/renamed-away files.
    for path in sorted(deleted):
        store.delete_by_file(path)

    # 2. Skip files whose content hash matches what's already indexed
    #    (avoids re-embedding on no-op merges or metadata-only changes).
    to_process = []
    for rel_path, path in added_or_modified.items():
        new_hash = file_hash(path)
        if store.get_file_hash(rel_path) == new_hash:
            continue
        to_process.append((rel_path, path, new_hash))

    if not to_process:
        print(f"[qatool] nothing to re-embed ({len(deleted)} deletions applied)")
        return

    # 3. Drop old chunks for files that changed, then re-chunk + re-embed.
    all_chunks = []
    for rel_path, full_path, new_hash in to_process:
        chunks = chunk_file(full_path, rel_path)  # -> list[Chunk(text, metadata)]
        all_chunks.append((rel_path, chunks, new_hash))

    total_chunks = sum(len(chunks) for _, chunks, _ in all_chunks)
    chunk_groups = [chunks for _, chunks, _ in all_chunks if chunks]
    if chunk_groups:
        flattened_chunks = [chunk for chunks in chunk_groups for chunk in chunks]
        vectors = embed_texts([chunk.text for chunk in flattened_chunks])
        if len(vectors) != len(flattened_chunks):
            raise ValueError(
                f"embedding count mismatch: expected {len(flattened_chunks)}, got {len(vectors)}"
            )

        offset = 0
        for rel_path, chunks, new_hash in all_chunks:
            chunk_vectors = vectors[offset:offset + len(chunks)]
            store.replace_file(rel_path, new_hash, chunks, chunk_vectors)
            offset += len(chunks)
    else:
        for rel_path, chunks, new_hash in all_chunks:
            store.replace_file(rel_path, new_hash, chunks, [])

    print(
        f"[qatool] reindexed {len(to_process)} file(s), "
        f"{total_chunks} chunk(s), {len(deleted)} deletion(s) applied"
    )


def require_existing_index(repo_root: Path) -> Path:
    index_path = repo_root / ".qatool" / "index" / VectorStore.STORE_FILENAME
    if not index_path.exists():
        raise ValueError(
            f"local index not found at {index_path}; run `qatool index {repo_root}` first"
        )
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument(
        "--changed-files",
        required=True,
        type=Path,
        help="Path to a file containing `git diff --name-status` output",
    )
    args = parser.parse_args()
    try:
        repo_root = validate_repository_path(args.repo)
        require_existing_index(repo_root)
        diff_text = args.changed_files.read_text()
    except (OSError, ValueError) as exc:
        print(f"[qatool] error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    changes = parse_diff_output(diff_text)
    if not changes:
        sys.exit(0)

    store = VectorStore.open(repo_root / ".qatool" / "index")
    reindex(repo_root, changes, store)


if __name__ == "__main__":
    main()
