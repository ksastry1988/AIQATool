"""
Exact symbol and filename lookup.

Embedding search tends to miss identifier-style questions ("where is
`replace_file` called?"), so `ask` also runs an exact-match pass over the
indexed chunk metadata and merges those hits ahead of the vector results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_BACKTICK_SPAN = re.compile(r"`([^`]+)`")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FILENAME = re.compile(r"(?:[\w.-]+/)*(?:[\w.-]*\w\.[A-Za-z][A-Za-z0-9]*|\.[\w-]+)")
_EDGE_PUNCTUATION = "\"'`,;:!?[]{}<>"
# Suffixes that make a bare dotted token (no `/`) a filename rather than a
# code reference: `store.json` is a file, `VectorStore.query` is an attribute.
_FILE_EXTENSIONS = frozenset(
    {
        "py", "pyi", "ipynb", "js", "jsx", "mjs", "cjs", "ts", "tsx", "go", "java",
        "kt", "kts", "scala", "rs", "rb", "php", "swift", "c", "cc", "cpp", "cxx", "h",
        "hh", "hpp", "cs", "m", "mm", "sh", "bash", "zsh", "ps1", "sql", "proto",
        "graphql", "json", "jsonl", "yaml", "yml", "toml", "ini", "cfg", "conf", "env",
        "xml", "html", "htm", "css", "scss", "sass", "less", "vue", "svelte", "md", "rst",
        "txt", "csv", "tsv", "lock", "log", "gradle", "properties", "tf", "dockerfile",
        "example", "sample", "template", "tmpl", "j2", "in", "mk", "cmake",
    }
)


@dataclass
class QueryTerms:
    symbols: set[str] = field(default_factory=set)
    filenames: set[str] = field(default_factory=set)


def _looks_like_code(name: str) -> bool:
    """True for names unlikely to be plain English: snake_case, camelCase, HTTPError."""
    if len(name) < 2:
        return False
    if "_" in name and name.strip("_"):
        return True
    return bool(re.search(r"[a-z][A-Z]|[A-Z]{2}[a-z]", name))


def _add_token(raw: str, terms: QueryTerms, explicit: bool) -> None:
    token = raw.strip(_EDGE_PUNCTUATION).rstrip(".")
    is_call = token.endswith("()")
    token = token.strip("()").rstrip(".").removeprefix("./")
    if not token:
        return

    if not is_call and _is_filename(token):
        terms.filenames.add(token)
        return

    parts = token.split(".")
    # In a dotted reference (`VectorStore.query`, `store.query`) the final
    # part is the member being asked about, even if it's a plain word.
    dotted = len(parts) > 1 and all(_IDENTIFIER.fullmatch(part) for part in parts)
    for index, part in enumerate(parts):
        if not _IDENTIFIER.fullmatch(part):
            continue
        is_member = dotted and index == len(parts) - 1 and len(part) >= 2
        if explicit or is_call or is_member or _looks_like_code(part):
            terms.symbols.add(part)


def _is_filename(token: str) -> bool:
    """Paths and dotfiles always count; a bare dotted token needs a known extension."""
    if not _FILENAME.fullmatch(token):
        return False
    if "/" in token or (token.startswith(".") and token.count(".") == 1):
        return True
    return token.rsplit(".", 1)[1].lower() in _FILE_EXTENSIONS


def extract_query_terms(question: str) -> QueryTerms:
    """Pull likely identifiers and filenames out of a natural-language question.

    Filenames are paths, dotfiles, or tokens with a known file extension
    (`qatool/cli.py`, `.qatoolignore`, `store.json`). Identifiers are
    backticked names, calls (`embed_texts()`), or code-shaped names
    (snake_case, camelCase); dotted references like `VectorStore.query`
    contribute their code-shaped parts plus the final member name.
    Ordinary words are ignored so prose doesn't match symbols by accident.
    """
    terms = QueryTerms()
    for span in _BACKTICK_SPAN.findall(question):
        for raw in span.split():
            _add_token(raw, terms, explicit=True)
    for raw in _BACKTICK_SPAN.sub(" ", question).split():
        _add_token(raw, terms, explicit=False)
    return terms


def _result_key(result: dict[str, Any]) -> Any:
    if result.get("id"):
        return result["id"]
    metadata = result.get("metadata", {})
    return (
        metadata.get("file"),
        metadata.get("start_line"),
        metadata.get("end_line"),
        metadata.get("symbol"),
        result.get("text", ""),
    )


def merge_results(
    exact: list[dict[str, Any]],
    semantic: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Combine exact-match and vector results into at most top_k chunks.

    Ranking: exact matches as ordered by `VectorStore.find_exact` (symbol,
    then definition, then filename), then vector results by similarity. Exact
    matches therefore survive even when their semantic score is low. A chunk
    found by both passes appears once, keeping its exact-match label.
    """
    merged: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for result in [*exact, *semantic]:
        key = _result_key(result)
        if key in seen:
            continue
        seen.add(key)
        merged.append(result)
        if len(merged) >= top_k:
            break
    return merged
