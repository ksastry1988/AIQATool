import json
import sys
from pathlib import Path

import pytest

from qatool.cli import main as cli_main
from qatool.chunking import Chunk
from qatool.indexing import index_repository
from qatool.vectorstore import VectorStore


def test_index_repository_respects_ignore_rules_and_secret_files(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "keep.py").write_text("def keep():\n    return 1\n")
    (repo / "ignored.py").write_text("def ignored():\n    return 2\n")
    (repo / ".env").write_text("API_KEY=secret")
    (repo / ".qatoolignore").write_text("ignored.py\n")

    captured_inputs: list[list[str]] = []

    def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        captured_inputs.append(list(texts))
        return [[float(len(text))] for text in texts]

    monkeypatch.setattr("qatool.indexing.embed_texts", fake_embed_texts)

    store = VectorStore.open(repo / ".qatool" / "index")
    index_repository(repo, store)

    persisted = json.loads((repo / ".qatool" / "index" / "store.json").read_text())
    assert set(persisted["file_hashes"]) == {"keep.py"}
    assert {chunk["metadata"]["file"] for chunk in persisted["chunks"]} == {"keep.py"}
    assert captured_inputs == [["def keep():\n    return 1\n"]]


def test_index_repository_skips_unchanged_files_on_repeat_runs(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "keep.py").write_text("def keep():\n    return 1\n")

    calls: list[list[str]] = []

    def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[1.0] for _ in texts]

    monkeypatch.setattr("qatool.indexing.embed_texts", fake_embed_texts)

    index_repository(repo, VectorStore.open(repo / ".qatool" / "index"))
    index_repository(repo, VectorStore.open(repo / ".qatool" / "index"))

    output = capsys.readouterr().out
    assert len(calls) == 1
    assert "1 unchanged" in output


def test_index_repository_batches_embeddings_and_reports_progress(
    tmp_path, monkeypatch, capsys
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def a():\n    return 'a'\n")
    (repo / "b.py").write_text("def b():\n    return 'b'\n")

    calls: list[list[str]] = []

    def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[float(index)] for index, _ in enumerate(texts, start=1)]

    monkeypatch.setattr("qatool.indexing.embed_texts", fake_embed_texts)

    index_repository(repo, VectorStore.open(repo / ".qatool" / "index"))

    output = capsys.readouterr().out
    assert calls == [["def a():\n    return 'a'\n", "def b():\n    return 'b'\n"]]
    assert "[qatool] indexing 1/2: a.py" in output
    assert "[qatool] indexing 2/2: b.py" in output


def test_index_repository_removes_deleted_files_from_store(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    file_path = repo / "keep.py"
    file_path.write_text("def keep():\n    return 1\n")

    store_path = repo / ".qatool" / "index"
    index_repository(repo, VectorStore.open(store_path))
    file_path.unlink()
    index_repository(repo, VectorStore.open(store_path))

    persisted = json.loads((store_path / "store.json").read_text())
    assert persisted["file_hashes"] == {}
    assert persisted["chunks"] == []


def test_vector_store_persists_and_queries_chunks(tmp_path):
    store = VectorStore.open(tmp_path / "index")
    store.upsert(
        [Chunk(text="alpha", metadata={"file": "a.py"})],
        [[1.0, 0.0]],
    )
    store.set_file_hash("a.py", "hash-a")

    reopened = VectorStore.open(tmp_path / "index")
    results = reopened.query([1.0, 0.0], top_k=1)

    assert reopened.get_file_hash("a.py") == "hash-a"
    assert results[0]["metadata"]["file"] == "a.py"
    assert results[0]["text"] == "alpha"


def test_vector_store_upsert_replaces_existing_file_chunks(tmp_path):
    store = VectorStore.open(tmp_path / "index")
    store.upsert(
        [Chunk(text="alpha", metadata={"file": "a.py"})],
        [[1.0]],
    )
    store.upsert(
        [Chunk(text="beta", metadata={"file": "a.py"})],
        [[2.0]],
    )

    persisted = json.loads((tmp_path / "index" / "store.json").read_text())
    assert [chunk["text"] for chunk in persisted["chunks"]] == ["beta"]


def test_index_repository_preserves_previous_data_when_embeddings_are_invalid(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    file_path = repo / "keep.py"
    file_path.write_text("def keep():\n    return 1\n")

    store_path = repo / ".qatool" / "index"
    index_repository(repo, VectorStore.open(store_path))
    previous = json.loads((store_path / "store.json").read_text())

    file_path.write_text("def keep():\n    return 2\n")

    def invalid_embed_texts(texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts[:-1]]

    monkeypatch.setattr("qatool.indexing.embed_texts", invalid_embed_texts)

    index_repository(repo, VectorStore.open(store_path))

    current = json.loads((store_path / "store.json").read_text())
    assert current == previous


def test_index_repository_continues_after_batch_embedding_failure(
    tmp_path, monkeypatch, capsys
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def good():\n    return 'ok'\n")
    (repo / "b.py").write_text("def bad():\n    return 'bad'\n")

    calls: list[list[str]] = []

    def flaky_embed_texts(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        if len(texts) > 1:
            raise RuntimeError("batch provider outage")
        if "bad" in texts[0]:
            raise RuntimeError("provider rejected content")
        return [[1.0]]

    monkeypatch.setattr("qatool.indexing.embed_texts", flaky_embed_texts)

    store_path = repo / ".qatool" / "index"
    index_repository(repo, VectorStore.open(store_path))

    persisted = json.loads((store_path / "store.json").read_text())
    error_output = capsys.readouterr().err
    assert calls[0] == ["def good():\n    return 'ok'\n", "def bad():\n    return 'bad'\n"]
    assert set(persisted["file_hashes"]) == {"a.py"}
    assert {chunk["metadata"]["file"] for chunk in persisted["chunks"]} == {"a.py"}
    assert "b.py: failed to index file (provider rejected content)" in error_output


def test_vector_store_rejects_query_dimension_mismatches(tmp_path):
    store = VectorStore.open(tmp_path / "index")
    store.upsert(
        [Chunk(text="alpha", metadata={"file": "a.py"})],
        [[1.0, 0.0]],
    )

    try:
        store.query([1.0], top_k=1)
    except ValueError as exc:
        assert "dimension mismatch" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("expected query to reject mismatched vector dimensions")


def test_vector_store_multiple_instances_preserve_writes(tmp_path):
    store_path = tmp_path / "index"
    primary = VectorStore.open(store_path)
    secondary = VectorStore.open(store_path)

    primary.set_file_hash("a.py", "hash-a")
    secondary.set_file_hash("b.py", "hash-b")

    reopened = VectorStore.open(store_path)
    assert reopened.get_file_hash("a.py") == "hash-a"
    assert reopened.get_file_hash("b.py") == "hash-b"


def test_vector_store_replace_file_updates_chunks_and_hash_atomically(tmp_path):
    store = VectorStore.open(tmp_path / "index")
    store.replace_file(
        "a.py",
        "hash-a",
        [Chunk(text="alpha", metadata={"file": "a.py"})],
        [[1.0]],
    )
    store.replace_file(
        "a.py",
        "hash-b",
        [Chunk(text="beta", metadata={"file": "a.py"})],
        [[2.0]],
    )

    reopened = VectorStore.open(tmp_path / "index")
    persisted = json.loads((tmp_path / "index" / "store.json").read_text())
    assert reopened.get_file_hash("a.py") == "hash-b"
    assert [chunk["text"] for chunk in persisted["chunks"]] == ["beta"]


def test_vector_store_cleans_temp_files_when_replace_fails(tmp_path, monkeypatch):
    store = VectorStore.open(tmp_path / "index")
    original_replace = Path.replace

    def failing_replace(self, target):
        if self.suffix == ".tmp":
            raise OSError("replace failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", failing_replace)

    try:
        store.set_file_hash("a.py", "hash-a")
    except OSError:
        pass
    else:  # pragma: no cover - assertion branch
        raise AssertionError("expected replace failure")

    assert list((tmp_path / "index").glob("*.tmp")) == []


def test_cli_index_rejects_missing_repository_without_creating_store(
    tmp_path, monkeypatch, capsys
):
    missing_repo = tmp_path / "missing-repo"
    monkeypatch.setattr(sys, "argv", ["qatool", "index", str(missing_repo)])

    with pytest.raises(SystemExit) as exc_info:
        cli_main()

    assert exc_info.value.code == 2
    assert f"repository does not exist: {missing_repo}" in capsys.readouterr().err
    assert not missing_repo.exists()
