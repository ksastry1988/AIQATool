"""
qatool CLI

    qatool index <repo>            Full index of a repository
    qatool reindex --repo <repo> --changed-files <diff-file>
    qatool ask "<question>"        Query the indexed codebase
"""

import argparse
import sys
from pathlib import Path


def cmd_index(args: argparse.Namespace) -> None:
    from .indexing import index_repository, validate_repository_path
    from .vectorstore import VectorStore

    repo = validate_repository_path(args.repo)
    store = VectorStore.open(repo / ".qatool" / "index")
    index_repository(repo, store)


def cmd_reindex(args: argparse.Namespace) -> None:
    from .reindex import main as reindex_main
    sys.argv = [
        "qatool-reindex",
        "--repo", str(args.repo),
        "--changed-files", str(args.changed_files),
    ]
    reindex_main()


def cmd_ask(args: argparse.Namespace) -> None:
    # TODO: embed query, retrieve top-k chunks, call Claude with context
    print(f"[qatool] question: {args.question!r} — not yet implemented")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qatool")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Full index of a repository")
    p_index.add_argument("repo", type=Path)
    p_index.set_defaults(func=cmd_index)

    p_reindex = sub.add_parser("reindex", help="Incremental re-index (used by git hook)")
    p_reindex.add_argument("--repo", required=True, type=Path)
    p_reindex.add_argument("--changed-files", required=True, type=Path)
    p_reindex.set_defaults(func=cmd_reindex)

    p_ask = sub.add_parser("ask", help="Ask a question about the codebase")
    p_ask.add_argument("question")
    p_ask.set_defaults(func=cmd_ask)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except ValueError as exc:
        parser.exit(2, f"[qatool] error: {exc}\n")


if __name__ == "__main__":
    main()
