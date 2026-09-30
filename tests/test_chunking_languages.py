"""AST-aware chunking against representative source files per language."""

from pathlib import Path

import pytest

from qatool import chunking
from qatool.chunking import chunk_file

FIXTURES = Path(__file__).parent / "fixtures" / "chunking"


def _chunks(name):
    return chunk_file(FIXTURES / name, name)


def _spans(chunks):
    return [(c.metadata["start_line"], c.metadata["end_line"], c.metadata["symbol"]) for c in chunks]


def _assert_exact_coverage(path, chunks):
    """Chunk text is exactly its lines, and every non-blank line is in one chunk."""
    lines = path.read_text().split("\n")
    seen = set()
    for chunk in chunks:
        start, end = chunk.metadata["start_line"], chunk.metadata["end_line"]
        expected = "\n".join(lines[start - 1:end]) + ("\n" if end < len(lines) else "")
        assert chunk.text == expected
        rows = set(range(start, end + 1))
        assert not rows & seen, f"overlapping lines {sorted(rows & seen)}"
        seen |= rows
    missing = [n for n, line in enumerate(lines, start=1) if line.strip() and n not in seen]
    assert not missing, f"lines not in any chunk: {missing}"


EXPECTED = {
    "sample.py": [(1, 5, None), (8, 14, "Bucket"), (17, 18, "refill"), (21, 22, None)],
    "sample.js": [
        (1, 1, None), (3, 6, "handle"), (8, 10, "retry"), (12, 12, None),
        (14, 18, "Client"), (20, 20, None),
    ],
    "sample.ts": [
        (1, 3, "Config"), (5, 5, "Mode"), (7, 10, "Level"),
        (12, 12, "makeConfig"), (14, 16, "Store"),
    ],
    "sample.tsx": [(1, 1, None), (3, 5, "Button")],
    "sample.go": [(1, 3, None), (5, 8, "Limiter"), (10, 13, "Allow"), (15, 17, "New")],
    "sample.java": [(1, 3, None), (5, 17, "Parser"), (19, 21, "Visitor")],
    "sample.rs": [(1, 1, None), (3, 8, "Point"), (10, 14, "Point"), (16, 18, "origin")],
    "sample.c": [
        (1, 3, None), (5, 8, "point"), (10, 12, "buffer"),
        (14, 16, "find_slot"), (18, 21, "main"),
    ],
    "sample.cpp": [(1, 1, None), (3, 17, "net")],
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_supported_languages_use_ast_chunks(name):
    chunks = _chunks(name)

    assert _spans(chunks) == EXPECTED[name]
    assert {c.metadata["language"] for c in chunks} == {name.rsplit(".", 1)[1]}
    assert {c.metadata["file"] for c in chunks} == {name}
    _assert_exact_coverage(FIXTURES / name, chunks)


def test_leading_comments_decorators_and_attributes_stay_with_definition():
    bucket = next(c for c in _chunks("sample.py") if c.metadata["symbol"] == "Bucket")
    point = next(c for c in _chunks("sample.rs") if c.metadata["symbol"] == "Point")
    allow = next(c for c in _chunks("sample.go") if c.metadata["symbol"] == "Allow")

    assert bucket.text.startswith("# Token bucket used by the API layer.\n@dataclass\n")
    assert point.text.startswith("/// A point in 2D space.\n#[derive(Debug, Clone)]\n")
    assert allow.text.startswith("// Allow reports whether a request may proceed.\n")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "sample.py",
            [(1, 3, None), (5, 5, None), (8, 11, "Bucket"), (13, 14, "take"),
             (17, 18, "refill"), (21, 22, None)],
        ),
        (
            "sample.java",
            [(1, 3, None), (5, 7, "Parser"), (9, 11, "Parser"), (13, 17, "toString"),
             (19, 21, "Visitor")],
        ),
        (
            "sample.rs",
            [(1, 1, None), (3, 8, "Point"), (10, 10, "Point"), (11, 14, "fmt"),
             (16, 18, "origin")],
        ),
        (
            "sample.cpp",
            [(1, 1, None), (3, 3, "net"), (5, 8, "Socket"), (10, 13, "clamp"),
             (15, 15, "open"), (17, 17, "net")],
        ),
    ],
)
def test_large_containers_split_into_members(monkeypatch, name, expected):
    monkeypatch.setattr(chunking, "MAX_CHUNK_LINES", 4)

    chunks = _chunks(name)

    assert _spans(chunks) == expected
    _assert_exact_coverage(FIXTURES / name, chunks)


def test_split_container_folds_closing_brace_into_last_member(monkeypatch):
    monkeypatch.setattr(chunking, "MAX_CHUNK_LINES", 4)

    to_string = next(c for c in _chunks("sample.java") if c.metadata["symbol"] == "toString")

    assert to_string.text.endswith("    }\n}\n")


def test_long_top_level_code_is_split_at_blank_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(chunking, "MAX_CHUNK_LINES", 4)
    path = tmp_path / "consts.py"
    path.write_text("A = 1\nB = 2\n\nC = 3\nD = 4\nE = 5\n\ndef f():\n    pass\n")

    chunks = chunk_file(path, "consts.py")

    assert _spans(chunks) == [(1, 2, None), (4, 6, None), (8, 9, "f")]
    _assert_exact_coverage(path, chunks)


def test_syntax_errors_still_produce_ast_chunks_covering_the_file(tmp_path):
    path = tmp_path / "broken.py"
    path.write_text("import os\n\ndef broken(:\n    pass\n\nclass Fine:\n    x = 1\n")

    chunks = chunk_file(path, "broken.py")

    assert "Fine" in [c.metadata["symbol"] for c in chunks]
    _assert_exact_coverage(path, chunks)


def test_parser_failure_falls_back_to_whole_file(tmp_path, monkeypatch, capsys):
    class ExplodingParser:
        def parse(self, source):
            raise RuntimeError("boom")

    monkeypatch.setattr(chunking, "_parser_for", lambda spec: ExplodingParser())
    path = tmp_path / "a.py"
    path.write_text("def f():\n    pass\n")

    chunks = chunk_file(path, "a.py")

    assert _spans(chunks) == [(1, 3, None)]
    assert chunks[0].text == "def f():\n    pass\n"
    assert "tree-sitter parse error in a.py: boom" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["notes.md", "script.rb", "Makefile", "data.json"])
def test_unsupported_languages_fall_back_to_whole_file(tmp_path, name):
    path = tmp_path / name
    path.write_text("def looks_like_code():\n    pass\n")

    chunks = chunk_file(path, name)

    assert len(chunks) == 1
    assert chunks[0].text == "def looks_like_code():\n    pass\n"
    assert chunks[0].metadata["symbol"] is None


def test_crlf_line_endings_keep_line_numbers(tmp_path):
    path = tmp_path / "win.py"
    path.write_bytes(b"import os\r\n\r\ndef f():\r\n    return 1\r\n")

    chunks = chunk_file(path, "win.py")

    # read_text() applies universal newlines, as for whole-file chunks.
    assert _spans(chunks) == [(1, 1, None), (3, 4, "f")]
    assert chunks[1].text == "def f():\n    return 1\n"
