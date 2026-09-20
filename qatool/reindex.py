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
from .indexing import file_hash, load_ignore_patterns, should_index
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

    added_or_modified: list[Path] = []
    deleted: list[str] = []

    for change in changes:
        if change.status == "D":
            deleted.append(change.path)
            continue

        target = change.new_path if change.status == "R" else change.path
        full_path = repo_root / target

        if change.status == "R" and change.path != change.new_path:
            # Old path's chunks are stale regardless of content.
            deleted.append(change.path)

        if not full_path.exists():
            continue  # file was deleted after the diff was captured
        if not should_index(full_path, repo_root, ignore_patterns):
            continue

        added_or_modified.append(full_path)

    # 1. Remove stale chunks for deleted/renamed-away files.
    for path in deleted:
        store.delete_by_file(path)

    # 2. Skip files whose content hash matches what's already indexed
    #    (avoids re-embedding on no-op merges or metadata-only changes).
    to_process = []
    for path in added_or_modified:
        rel_path = path.relative_to(repo_root).as_posix()
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
        store.delete_by_file(rel_path)
        chunks = chunk_file(full_path, rel_path)  # -> list[Chunk(text, metadata)]
        all_chunks.extend(chunks)
        store.set_file_hash(rel_path, new_hash)

    if all_chunks:
        vectors = embed_texts([c.text for c in all_chunks])
        store.upsert(chunks=all_chunks, vectors=vectors)

    print(
        f"[qatool] reindexed {len(to_process)} file(s), "
        f"{len(all_chunks)} chunk(s), {len(deleted)} deletion(s) applied"
    )


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

    diff_text = args.changed_files.read_text()
    changes = parse_diff_output(diff_text)
    if not changes:
        sys.exit(0)

    store = VectorStore.open(args.repo / ".qatool" / "index")
    reindex(args.repo, changes, store)


if __name__ == "__main__":
    main()
