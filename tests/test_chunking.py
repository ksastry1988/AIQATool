"""Tests for qatool.chunking module."""

import tempfile
from pathlib import Path

import pytest

from qatool.chunking import Chunk, chunk_file


class TestChunkFile:
    """Tests for chunk_file function."""

    def test_chunk_file_returns_list(self):
        """chunk_file should return a list of Chunk objects."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("sample text")
            f.flush()
            path = Path(f.name)

        try:
            chunks = chunk_file(path, "sample.txt")
            assert isinstance(chunks, list)
            assert len(chunks) > 0
            assert all(isinstance(c, Chunk) for c in chunks)
        finally:
            path.unlink()

    def test_chunk_metadata_includes_required_fields(self):
        """Chunk metadata must include file, start_line, end_line, language."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello():\n    return 'world'")
            f.flush()
            path = Path(f.name)

        try:
            chunks = chunk_file(path, "hello.py")
            assert len(chunks) >= 1
            chunk = chunks[0]
            assert chunk.metadata["file"] == "hello.py"
            assert "start_line" in chunk.metadata
            assert "end_line" in chunk.metadata
            assert chunk.metadata["language"] == "py"
        finally:
            path.unlink()

    def test_chunk_empty_file(self):
        """Empty file should produce a single empty chunk."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("")
            f.flush()
            path = Path(f.name)

        try:
            chunks = chunk_file(path, "empty.py")
            assert len(chunks) == 1
            assert chunks[0].text == ""
        finally:
            path.unlink()

    def test_chunk_preserves_file_content(self):
        """Chunking should preserve all file content."""
        content = "line 1\nline 2\nline 3\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            chunks = chunk_file(path, "test.txt")
            reconstructed = "".join(c.text for c in chunks)
            assert reconstructed == content
        finally:
            path.unlink()

    def test_chunk_python_file_without_tree_sitter(self):
        """Python file should still chunk even without tree-sitter."""
        python_code = """def func1():
    pass

def func2():
    pass

class MyClass:
    def method(self):
        pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(python_code)
            f.flush()
            path = Path(f.name)

        try:
            chunks = chunk_file(path, "test.py")
            # Without tree-sitter, should fallback to whole-file
            assert len(chunks) >= 1
            assert chunks[0].text == python_code
            assert chunks[0].metadata["symbol"] is None
        finally:
            path.unlink()

    def test_chunk_javascript_file(self):
        """JavaScript file should chunk correctly."""
        js_code = """function hello() {
  return 'world';
}

function goodbye() {
  return 'farewell';
}

class Greeter {
  greet() {
    return 'hi';
  }
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(js_code)
            f.flush()
            path = Path(f.name)

        try:
            chunks = chunk_file(path, "test.js")
            assert len(chunks) >= 1
            # Check that first chunk contains valid text
            assert chunks[0].text
            assert chunks[0].metadata["language"] == "js"
        finally:
            path.unlink()

    def test_chunk_line_numbers_correct(self):
        """Line numbers in metadata should match actual content."""
        content = "line 1\nline 2\nline 3\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            chunks = chunk_file(path, "test.txt")
            chunk = chunks[0]
            # File has 3 lines
            assert chunk.metadata["end_line"] >= chunk.metadata["start_line"]
        finally:
            path.unlink()

    def test_chunk_handles_binary_files(self):
        """Binary files should be handled gracefully (errors='ignore')."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n")  # PNG header
            f.flush()
            path = Path(f.name)

        try:
            # Should not raise
            chunks = chunk_file(path, "test.bin")
            assert len(chunks) >= 1
        finally:
            path.unlink()

    def test_chunk_multiline_content(self):
        """Multiline files should preserve all lines in chunks."""
        lines = [f"line {i}\n" for i in range(1, 11)]
        content = "".join(lines)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            chunks = chunk_file(path, "test.txt")
            reconstructed = "".join(c.text for c in chunks)
            assert reconstructed == content
            assert reconstructed.count("\n") == 10
        finally:
            path.unlink()

    def test_chunk_unicode_content(self):
        """Unicode content should be preserved correctly."""
        content = "Hello 世界\nПривет мир\n🚀 Rocket\n"
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", 
                                        delete=False, encoding="utf-8") as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            chunks = chunk_file(path, "test.txt")
            reconstructed = "".join(c.text for c in chunks)
            assert reconstructed == content
        finally:
            path.unlink()

    def test_chunk_rel_path_in_metadata(self):
        """Relative path should be stored correctly in metadata."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("test")
            f.flush()
            path = Path(f.name)

        try:
            rel_path = "src/utils/helpers.py"
            chunks = chunk_file(path, rel_path)
            assert chunks[0].metadata["file"] == rel_path
        finally:
            path.unlink()

    def test_chunk_language_detection(self):
        """Language should be detected from file extension."""
        test_cases = [
            ("test.py", "py"),
            ("test.js", "js"),
            ("test.ts", "ts"),
            ("test.go", "go"),
            ("test.rs", "rs"),
            ("test.java", "java"),
        ]
        
        for filename, expected_lang in test_cases:
            with tempfile.NamedTemporaryFile(mode="w", suffix=filename, delete=False) as f:
                f.write("content")
                f.flush()
                path = Path(f.name)

            try:
                chunks = chunk_file(path, filename)
                assert chunks[0].metadata["language"] == expected_lang
            finally:
                path.unlink()
