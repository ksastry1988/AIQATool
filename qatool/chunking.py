"""
Language-aware chunking.

Splits a source file into semantically coherent chunks (functions,
classes, top-level blocks) rather than fixed-size windows, so retrieval
returns whole units of meaning instead of arbitrary line slices.

Grammars come from the per-language `tree-sitter-<lang>` packages. Files
in other languages, files whose grammar package is missing, and files
that fail to parse fall back to a single whole-file chunk.

AST chunking rules:
- Each top-level definition becomes one chunk, including its decorators,
  `export`/`template` wrappers, and directly preceding comments or
  attributes.
- A container (class, impl, trait, namespace, ...) longer than
  MAX_CHUNK_LINES is split into its member definitions plus chunks for
  the code between them (header, fields), labelled with the container's
  name. Smaller containers stay whole.
- Code between definitions (imports, constants, statements) becomes its
  own chunks, split at blank lines to stay within MAX_CHUNK_LINES, so
  every non-blank line lands in exactly one chunk. Definitions themselves
  are never split below the container level.
- Chunks are whole lines: `text` is exactly lines start_line..end_line.
- Syntax errors don't abort chunking; tree-sitter recovers and the rules
  above still cover every line.
"""

from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:  # pragma: no cover - tree-sitter is a declared dependency
    TREE_SITTER_AVAILABLE = False

MAX_CHUNK_LINES = 80


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _LanguageSpec:
    module: str                     # grammar package, e.g. tree_sitter_python
    function: str                   # its language() entry point
    definitions: frozenset[str]     # node types that become chunks
    containers: frozenset[str] = frozenset()  # definitions split when large


_PYTHON = _LanguageSpec(
    "tree_sitter_python",
    "language",
    frozenset({"function_definition", "class_definition", "decorated_definition"}),
    frozenset({"class_definition"}),
)
_JS_DEFINITIONS = frozenset(
    {
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "method_definition",
        "lexical_declaration",
        "variable_declaration",
        "export_statement",
    }
)
_JAVASCRIPT = _LanguageSpec(
    "tree_sitter_javascript", "language", _JS_DEFINITIONS, frozenset({"class_declaration"})
)
_TS_DEFINITIONS = _JS_DEFINITIONS | {
    "abstract_class_declaration",
    "interface_declaration",
    "type_alias_declaration",
    "enum_declaration",
}
_TS_CONTAINERS = frozenset({"class_declaration", "abstract_class_declaration"})
_TYPESCRIPT = _LanguageSpec(
    "tree_sitter_typescript", "language_typescript", _TS_DEFINITIONS, _TS_CONTAINERS
)
_TSX = _LanguageSpec("tree_sitter_typescript", "language_tsx", _TS_DEFINITIONS, _TS_CONTAINERS)
_GO = _LanguageSpec(
    "tree_sitter_go",
    "language",
    frozenset({"function_declaration", "method_declaration", "type_declaration"}),
)
_JAVA_CONTAINERS = frozenset(
    {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"}
)
_JAVA = _LanguageSpec(
    "tree_sitter_java",
    "language",
    _JAVA_CONTAINERS
    | {"annotation_type_declaration", "method_declaration", "constructor_declaration"},
    _JAVA_CONTAINERS,
)
_RUST = _LanguageSpec(
    "tree_sitter_rust",
    "language",
    frozenset(
        {
            "function_item",
            "function_signature_item",
            "struct_item",
            "enum_item",
            "union_item",
            "trait_item",
            "impl_item",
            "mod_item",
            "type_item",
            "macro_definition",
        }
    ),
    frozenset({"impl_item", "trait_item", "mod_item"}),
)
_C_DEFINITIONS = frozenset(
    {"function_definition", "struct_specifier", "union_specifier", "enum_specifier", "type_definition"}
)
_C = _LanguageSpec("tree_sitter_c", "language", _C_DEFINITIONS)
_CPP = _LanguageSpec(
    "tree_sitter_cpp",
    "language",
    _C_DEFINITIONS | {"class_specifier", "namespace_definition", "template_declaration"},
    frozenset({"class_specifier", "struct_specifier", "namespace_definition"}),
)

# File extension -> grammar. `language` metadata stays the bare extension.
LANGUAGES: dict[str, _LanguageSpec] = {
    "py": _PYTHON,
    "pyi": _PYTHON,
    "js": _JAVASCRIPT,
    "jsx": _JAVASCRIPT,
    "mjs": _JAVASCRIPT,
    "cjs": _JAVASCRIPT,
    "ts": _TYPESCRIPT,
    "mts": _TYPESCRIPT,
    "cts": _TYPESCRIPT,
    "tsx": _TSX,
    "go": _GO,
    "java": _JAVA,
    "rs": _RUST,
    "c": _C,
    "h": _C,
    "cc": _CPP,
    "cpp": _CPP,
    "cxx": _CPP,
    "hh": _CPP,
    "hpp": _CPP,
    "hxx": _CPP,
}

# Nodes wrapping the real definition: field holding it, or None to scan children.
_WRAPPERS = {
    "decorated_definition": "definition",
    "export_statement": "declaration",
    "template_declaration": None,
}
# Specifiers only define something when they have a body (`struct P *p;` doesn't).
_BODY_REQUIRED = {"struct_specifier", "union_specifier", "enum_specifier", "class_specifier"}
_FUNCTION_VALUES = {
    "arrow_function",
    "function_expression",
    "function",
    "generator_function",
    "class",
}
_LEADING_TRIVIA = {"comment", "line_comment", "block_comment", "attribute_item"}
_NAME_LEAVES = {
    "identifier",
    "field_identifier",
    "type_identifier",
    "destructor_name",
    "operator_name",
}
_CLOSING_ONLY = re.compile(r"[\s{}()\[\];,]*")


@lru_cache(maxsize=None)
def _parser_for(spec: _LanguageSpec) -> Any | None:
    """Load a parser for the grammar, or None if its package isn't installed."""
    if not TREE_SITTER_AVAILABLE:
        return None
    try:
        module = importlib.import_module(spec.module)
    except ImportError:
        return None
    return Parser(Language(getattr(module, spec.function)()))


def _unwrap(node: Any, spec: _LanguageSpec) -> Any | None:
    """Return the definition inside a wrapper node (or the node itself)."""
    if node.type not in _WRAPPERS:
        return node
    field_name = _WRAPPERS[node.type]
    if field_name is not None:
        inner = node.child_by_field_name(field_name)
        if inner is not None:
            return _unwrap(inner, spec) if _is_definition(inner, spec) else None
    for child in node.named_children:
        if child.type not in _WRAPPERS and _is_definition(child, spec):
            return child
    return None


def _function_declarator(node: Any) -> Any | None:
    """First `const f = () => ...`-style declarator in a JS/TS declaration."""
    for child in node.named_children:
        if child.type == "variable_declarator":
            value = child.child_by_field_name("value")
            if value is not None and value.type in _FUNCTION_VALUES:
                return child
    return None


def _is_definition(node: Any, spec: _LanguageSpec) -> bool:
    if node.type not in spec.definitions:
        return False
    if node.type in _WRAPPERS:
        return _unwrap(node, spec) is not None
    if node.type in ("lexical_declaration", "variable_declaration"):
        return _function_declarator(node) is not None
    if node.type in _BODY_REQUIRED:
        return node.child_by_field_name("body") is not None
    return True


def _leaf_name(node: Any) -> str:
    while node.type == "qualified_identifier" and node.child_by_field_name("name") is not None:
        node = node.child_by_field_name("name")
    return node.text.decode("utf-8", errors="replace")


def _symbol_name(node: Any, spec: _LanguageSpec) -> str | None:
    node = _unwrap(node, spec)
    if node is None:
        return None
    if node.type in ("lexical_declaration", "variable_declaration"):
        node = _function_declarator(node)
    elif node.type == "type_declaration":
        node = next((child for child in node.named_children if child.type.startswith("type_")), None)
    elif node.type == "impl_item":
        impl_type = node.child_by_field_name("type")
        return _leaf_name(impl_type) if impl_type is not None else None
    if node is None:
        return None

    name = node.child_by_field_name("name")
    if name is not None:
        return _leaf_name(name)

    # C/C++: the name sits at the bottom of a declarator chain.
    declarator = node.child_by_field_name("declarator")
    while declarator is not None:
        if declarator.type in _NAME_LEAVES or declarator.type == "qualified_identifier":
            return _leaf_name(declarator)
        declarator = declarator.child_by_field_name("declarator")
    return None


def _collect_definitions(node: Any, spec: _LanguageSpec) -> list[Any]:
    """Outermost definitions below `node`, without descending into them."""
    found = []
    for child in node.named_children:
        if _is_definition(child, spec):
            found.append(child)
        else:
            found.extend(_collect_definitions(child, spec))
    return found


def _row_span(node: Any) -> tuple[int, int]:
    """0-based inclusive rows, extended up over attached comments/attributes."""
    start, end = node.start_point[0], node.end_point[0]
    if node.end_point[1] == 0 and end > start:
        end -= 1  # node ends at the start of the next line

    previous = node.prev_named_sibling
    while previous is not None and previous.type in _LEADING_TRIVIA and previous.end_point[0] >= start - 1:
        start = previous.start_point[0]
        previous = previous.prev_named_sibling
    return start, end


class _ChunkBuilder:
    def __init__(self, text: str, rel_path: str, language: str, spec: _LanguageSpec) -> None:
        pieces = text.split("\n")
        self.lines = [piece + "\n" for piece in pieces[:-1]] + [pieces[-1]]
        self.rel_path = rel_path
        self.language = language
        self.spec = spec
        self.chunks: list[Chunk] = []

    def _emit(self, start: int, end: int, symbol: str | None) -> None:
        while start <= end and not self.lines[start].strip():
            start += 1
        while end >= start and not self.lines[end].strip():
            end -= 1
        if start > end:
            return

        text = "".join(self.lines[start:end + 1])
        previous = self.chunks[-1] if self.chunks else None
        if previous is not None and _CLOSING_ONLY.fullmatch(text):
            # A lone `}` / `};` after a split container: fold it into the
            # previous chunk rather than embedding punctuation.
            gap = "".join(self.lines[previous.metadata["end_line"]:start])
            previous.text += gap + text
            previous.metadata["end_line"] = end + 1
            return

        self.chunks.append(
            Chunk(
                text=text,
                metadata={
                    "file": self.rel_path,
                    "start_line": start + 1,
                    "end_line": end + 1,
                    "symbol": symbol,
                    "language": self.language,
                },
            )
        )

    def _emit_gap(self, start: int, end: int, symbol: str | None) -> None:
        """Emit non-definition code, split at blank lines to stay under MAX_CHUNK_LINES."""
        while end - start + 1 > MAX_CHUNK_LINES:
            limit = start + MAX_CHUNK_LINES - 1
            split = next((row for row in range(limit, start, -1) if not self.lines[row].strip()), limit)
            self._emit(start, split, symbol)
            start = split + 1
        self._emit(start, end, symbol)

    def region(self, start: int, end: int, definitions: list[Any], gap_symbol: str | None) -> None:
        """Chunk rows start..end: definitions as units, the rest as gap chunks."""
        cursor = start
        for node in sorted(definitions, key=lambda item: item.start_byte):
            node_start, node_end = _row_span(node)
            node_start = max(node_start, cursor)
            if node_end < node_start:
                continue  # shares a line already emitted with a previous chunk

            self._emit_gap(cursor, node_start - 1, gap_symbol)
            name = _symbol_name(node, self.spec)
            inner = _unwrap(node, self.spec)
            members = (
                _collect_definitions(inner, self.spec)
                if inner is not None and inner.type in self.spec.containers
                else []
            )
            if members and node_end - node_start + 1 > MAX_CHUNK_LINES:
                self.region(node_start, node_end, members, name)
            else:
                self._emit(node_start, node_end, name)
            cursor = node_end + 1
        self._emit_gap(cursor, end, gap_symbol)


def extract_chunks_tree_sitter(text: str, ext: str, rel_path: str) -> list[Chunk] | None:
    """
    Extract AST-aware chunks from text.

    Returns None when the extension is unsupported, its grammar can't be
    loaded, or parsing fails, so the caller can fall back to one chunk.
    """
    spec = LANGUAGES.get(ext.lstrip("."))
    if spec is None or not text.strip():
        return None
    parser = _parser_for(spec)
    if parser is None:
        return None

    try:
        tree = parser.parse(text.encode("utf-8"))
        builder = _ChunkBuilder(text, rel_path, ext, spec)
        builder.region(0, len(builder.lines) - 1, _collect_definitions(tree.root_node, spec), None)
    except Exception as exc:
        print(f"[qatool] tree-sitter parse error in {rel_path}: {exc}", file=sys.stderr)
        return None
    return builder.chunks or None


def chunk_file(full_path: Path, rel_path: str) -> list[Chunk]:
    """
    Return a list of Chunks for one file.

    Each chunk's metadata includes:
        - file: rel_path
        - start_line / end_line (1-based, inclusive)
        - symbol: function/class name, if applicable
        - language: the file extension

    Uses tree-sitter for semantic splitting; falls back to whole-file
    chunking for unsupported languages.
    """
    text = full_path.read_text(errors="ignore")
    ext = full_path.suffix.lstrip(".")

    chunks = extract_chunks_tree_sitter(text, ext, rel_path)
    if chunks is not None:
        return chunks

    return [
        Chunk(
            text=text,
            metadata={
                "file": rel_path,
                "start_line": 1,
                "end_line": text.count("\n") + 1,
                "symbol": None,
                "language": ext,
            },
        )
    ]
