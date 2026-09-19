"""
Language-aware chunking.

Splits a source file into semantically coherent chunks (functions,
classes, top-level blocks) rather than fixed-size windows, so retrieval
returns whole units of meaning instead of arbitrary line slices.

Uses tree-sitter for AST-aware parsing per language, with graceful
fallback to whole-file chunks for unsupported languages.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tree_sitter
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


# Map file extensions to tree-sitter language names
LANGUAGE_MAP = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "jsx": "javascript",
    "go": "go",
    "java": "java",
    "rs": "rust",
    "rb": "ruby",
    "c": "c",
    "cpp": "cpp",
    "h": "c",
    "hpp": "cpp",
}

# Query patterns per language to extract functions/classes
QUERIES = {
    "python": """
    (function_definition name: (identifier) @func.name) @func
    (class_definition name: (identifier) @class.name) @class
    """,
    "javascript": """
    (function_declaration name: (identifier) @func.name) @func
    (class_declaration name: (identifier) @class.name) @class
    (arrow_function) @func
    """,
    "typescript": """
    (function_declaration name: (identifier) @func.name) @func
    (class_declaration name: (identifier) @class.name) @class
    (arrow_function) @func
    """,
    "go": """
    (function_declaration name: (identifier) @func.name) @func
    (method_declaration name: (field_identifier) @method.name) @method
    """,
    "java": """
    (method_declaration name: (identifier) @method.name) @method
    (class_declaration name: (identifier) @class.name) @class
    """,
    "rust": """
    (function_item name: (identifier) @func.name) @func
    (struct_item name: (type_identifier) @struct.name) @struct
    (impl_item) @impl
    """,
}


def get_language(ext: str):
    """Get tree-sitter Language object for file extension, or None if unsupported."""
    if not TREE_SITTER_AVAILABLE:
        return None
    
    lang_name = LANGUAGE_MAP.get(ext.lstrip("."))
    if not lang_name:
        return None
    
    try:
        return Language(f"build/languages/tree-sitter-{lang_name}/language.so", lang_name)
    except Exception:
        return None


def extract_chunks_tree_sitter(
    text: str, language: str, rel_path: str
) -> list[Chunk] | None:
    """
    Extract semantic chunks from text using tree-sitter.
    Returns None if tree-sitter unavailable or unsupported language.
    """
    if not TREE_SITTER_AVAILABLE:
        return None
    
    lang_obj = get_language(language)
    if not lang_obj:
        return None
    
    try:
        parser = Parser()
        parser.set_language(lang_obj)
        tree = parser.parse(text.encode("utf-8"))
        
        chunks = []
        lines = text.splitlines(keepends=True)
        
        def extract_node(node):
            """Recursively extract function/class definitions."""
            # Check for function or class definition
            if node.type in ("function_definition", "function_declaration", 
                           "class_definition", "class_declaration", "method_declaration",
                           "arrow_function"):
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                
                # Extract symbol name
                symbol = None
                for child in node.children:
                    if child.type == "identifier" or child.type == "type_identifier":
                        symbol = child.text.decode("utf-8")
                        break
                
                chunk_text = "".join(lines[node.start_point[0]:node.end_point[0] + 1])
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            "file": rel_path,
                            "start_line": start_line,
                            "end_line": end_line,
                            "symbol": symbol,
                            "language": language,
                        },
                    )
                )
            
            # Recurse into children
            for child in node.children:
                extract_node(child)
        
        extract_node(tree.root_node)
        return chunks if chunks else None
    
    except Exception as e:
        print(f"[qatool] tree-sitter parse error in {rel_path}: {e}", file=sys.stderr)
        return None


def chunk_file(full_path: Path, rel_path: str) -> list[Chunk]:
    """
    Return a list of Chunks for one file.

    Each chunk's metadata includes:
        - file: rel_path
        - start_line / end_line
        - symbol: function/class name, if applicable
        - language

    Uses tree-sitter for semantic splitting; falls back to whole-file
    chunking for unsupported languages.
    """
    text = full_path.read_text(errors="ignore")
    ext = full_path.suffix.lstrip(".")
    
    # Try tree-sitter if available
    chunks = extract_chunks_tree_sitter(text, ext, rel_path)
    if chunks is not None:
        return chunks
    
    # Fallback: treat entire file as one chunk
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
