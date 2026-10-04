import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from qatool.chunking import Chunk
from qatool.indexing import file_hash
from qatool.reindex import main as reindex_main
from qatool.reindex import parse_diff_output
from qatool.reindex import reindex
from qatool.vectorstore import VectorStore


def test_parse_diff_output_handles_add_modify_delete():
    diff_text = "A\tnew_file.py\nM\tchanged_file.py\nD\tremoved_file.py"
    changes = parse_diff_output(diff_text)

    statuses = {c.path: c.status for c in changes}
    assert statuses["new_file.py"] == "A"
    assert statuses["changed_file.py"] == "M"
    assert statuses["removed_file.py"] == "D"


def test_parse_diff_output_handles_rename():
    diff_text = "R100\told_name.py\tnew_name.py"
    changes = parse_diff_output(diff_text)

    assert len(changes) == 1
    assert changes[0].status == "R"
    assert changes[0].path == "old_name.py"
    assert changes[0].new_path == "new_name.py"


def test_reindex_deduplicates_overlapping_changed_paths(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "tracked.py").write_text("print('one')\n")

    calls: list[str] = []

    def fake_chunk_file(full_path: Path, rel_path: str):
        calls.append(rel_path)
        return [Chunk(text=full_path.read_text(), metadata={"file": rel_path})]

    monkeypatch.setattr("qatool.reindex.chunk_file", fake_chunk_file)
    monkeypatch.setattr("qatool.reindex.embed_texts", lambda texts: [[1.0] for _ in texts])

    reindex(
        repo,
        parse_diff_output("M\ttracked.py\nM\ttracked.py\n"),
        VectorStore.open(repo / ".qatool" / "index"),
    )

    assert calls == ["tracked.py"]


def _seed_indexed_file(repo: Path, store: VectorStore, rel_path: str, content: str) -> Path:
    path = repo / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    store.replace_file(
        rel_path,
        file_hash(path),
        [Chunk(text=content, metadata={"file": rel_path})],
        [[1.0]],
    )
    return path


def _chunks_for_file(store: VectorStore, rel_path: str) -> list[dict]:
    return [
        chunk
        for chunk in store.query([1.0], top_k=100)
        if chunk["metadata"].get("file") == rel_path
    ]


def test_reindex_indexes_added_file(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    added_path = repo / "added.py"
    added_path.write_text("print('added')\n")
    store = VectorStore.open(repo / ".qatool" / "index")
    monkeypatch.setattr(
        "qatool.reindex.chunk_file",
        lambda full_path, rel_path: [
            Chunk(text=full_path.read_text(), metadata={"file": rel_path})
        ],
    )
    monkeypatch.setattr("qatool.reindex.embed_texts", lambda texts: [[1.0] for _ in texts])

    reindex(repo, parse_diff_output("A\tadded.py"), store)

    reopened = VectorStore.open(repo / ".qatool" / "index")
    assert reopened.get_file_hash("added.py") == file_hash(added_path)
    assert [chunk["text"] for chunk in _chunks_for_file(reopened, "added.py")] == [
        "print('added')\n"
    ]


def test_reindex_replaces_modified_file_chunks(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    store = VectorStore.open(repo / ".qatool" / "index")
    modified_path = _seed_indexed_file(repo, store, "modified.py", "print('old')\n")
    _seed_indexed_file(repo, store, "unchanged.py", "print('keep')\n")
    modified_path.write_text("print('new')\n")
    monkeypatch.setattr(
        "qatool.reindex.chunk_file",
        lambda full_path, rel_path: [
            Chunk(text=full_path.read_text(), metadata={"file": rel_path})
        ],
    )
    monkeypatch.setattr("qatool.reindex.embed_texts", lambda texts: [[1.0] for _ in texts])

    reindex(repo, parse_diff_output("M\tmodified.py"), store)

    reopened = VectorStore.open(repo / ".qatool" / "index")
    assert reopened.get_file_hash("modified.py") == file_hash(modified_path)
    assert [chunk["text"] for chunk in _chunks_for_file(reopened, "modified.py")] == [
        "print('new')\n"
    ]
    assert [chunk["text"] for chunk in _chunks_for_file(reopened, "unchanged.py")] == [
        "print('keep')\n"
    ]


def test_reindex_removes_deleted_file_chunks_and_hash(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    store = VectorStore.open(repo / ".qatool" / "index")
    deleted_path = _seed_indexed_file(repo, store, "deleted.py", "print('gone')\n")
    deleted_path.unlink()

    reindex(repo, parse_diff_output("D\tdeleted.py"), store)

    reopened = VectorStore.open(repo / ".qatool" / "index")
    assert reopened.get_file_hash("deleted.py") is None
    assert _chunks_for_file(reopened, "deleted.py") == []


def test_reindex_moves_renamed_file_chunks_and_hash(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    store = VectorStore.open(repo / ".qatool" / "index")
    old_path = _seed_indexed_file(repo, store, "old.py", "print('renamed')\n")
    new_path = repo / "new.py"
    old_path.rename(new_path)
    monkeypatch.setattr(
        "qatool.reindex.chunk_file",
        lambda full_path, rel_path: [
            Chunk(text=full_path.read_text(), metadata={"file": rel_path})
        ],
    )
    monkeypatch.setattr("qatool.reindex.embed_texts", lambda texts: [[1.0] for _ in texts])

    reindex(repo, parse_diff_output("R100\told.py\tnew.py"), store)

    reopened = VectorStore.open(repo / ".qatool" / "index")
    assert reopened.get_file_hash("old.py") is None
    assert _chunks_for_file(reopened, "old.py") == []
    assert reopened.get_file_hash("new.py") == file_hash(new_path)
    assert [chunk["text"] for chunk in _chunks_for_file(reopened, "new.py")] == [
        "print('renamed')\n"
    ]


def test_reindex_skips_unchanged_file_without_embedding(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    store = VectorStore.open(repo / ".qatool" / "index")
    path = _seed_indexed_file(repo, store, "unchanged.py", "print('same')\n")
    embedding_calls = []
    monkeypatch.setattr(
        "qatool.reindex.embed_texts",
        lambda texts: embedding_calls.append(texts),
    )

    reindex(repo, parse_diff_output("M\tunchanged.py"), store)

    reopened = VectorStore.open(repo / ".qatool" / "index")
    assert reopened.get_file_hash("unchanged.py") == file_hash(path)
    assert [chunk["text"] for chunk in _chunks_for_file(reopened, "unchanged.py")] == [
        "print('same')\n"
    ]
    assert embedding_calls == []


def _run_git(
    repo: Path,
    *args: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
):
    return subprocess.run(
        ["git", *args],
        cwd=cwd or repo,
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


def _setup_indexed_git_repo(
    tmp_path: Path, files: dict[str, str]
) -> tuple[Path, VectorStore, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init", "--initial-branch=master")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "user.email", "test@example.com")
    (repo / ".gitignore").write_text(".qatool/\n")
    for rel_path, content in files.items():
        path = repo / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-m", "base commit")
    base_sha = _run_git(repo, "rev-parse", "HEAD").stdout.strip()

    store = VectorStore.open(repo / ".qatool" / "index")
    for rel_path, content in files.items():
        _seed_indexed_file(repo, store, rel_path, content)
    store.set_indexed_commit_sha(base_sha)
    return repo, store, base_sha


def test_reindex_successfully_advances_commit_marker_with_file_changes(
    tmp_path, monkeypatch
):
    repo, store, base_sha = _setup_indexed_git_repo(
        tmp_path, {"tracked.py": "print('old')\n"}
    )
    tracked_path = repo / "tracked.py"
    tracked_path.write_text("print('new')\n")
    _run_git(repo, "add", "tracked.py")
    _run_git(repo, "commit", "-m", "update tracked file")
    target_sha = _run_git(repo, "rev-parse", "HEAD").stdout.strip()
    monkeypatch.setattr(
        "qatool.reindex.chunk_file",
        lambda full_path, rel_path: [
            Chunk(text=full_path.read_text(), metadata={"file": rel_path})
        ],
    )
    monkeypatch.setattr("qatool.reindex.embed_texts", lambda texts: [[2.0] for _ in texts])

    reindex(repo, parse_diff_output("M\ttracked.py"), store)

    reopened = VectorStore.open(repo / ".qatool" / "index")
    assert base_sha != target_sha
    assert reopened.get_indexed_commit_sha() == target_sha
    assert reopened.get_file_hash("tracked.py") == file_hash(tracked_path)
    assert [chunk["text"] for chunk in _chunks_for_file(reopened, "tracked.py")] == [
        "print('new')\n"
    ]


def test_reindex_embedding_failure_preserves_index_and_commit_marker(
    tmp_path, monkeypatch
):
    repo, store, base_sha = _setup_indexed_git_repo(
        tmp_path,
        {"tracked.py": "print('old')\n", "removed.py": "print('remove')\n"},
    )
    store_path = repo / ".qatool" / "index" / "store.json"
    previous_data = store_path.read_text()
    removed_hash = store.get_file_hash("removed.py")
    (repo / "tracked.py").write_text("print('new')\n")
    (repo / "removed.py").unlink()
    _run_git(repo, "add", "-A")
    _run_git(repo, "commit", "-m", "modify and delete")
    monkeypatch.setattr(
        "qatool.reindex.chunk_file",
        lambda full_path, rel_path: [
            Chunk(text=full_path.read_text(), metadata={"file": rel_path})
        ],
    )

    def fail_embedding(texts):
        raise RuntimeError("embedding unavailable")

    monkeypatch.setattr("qatool.reindex.embed_texts", fail_embedding)

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        reindex(repo, parse_diff_output("M\ttracked.py\nD\tremoved.py"), store)

    assert store_path.read_text() == previous_data
    reopened = VectorStore.open(repo / ".qatool" / "index")
    assert reopened.get_indexed_commit_sha() == base_sha
    assert reopened.get_file_hash("removed.py") == removed_hash


def test_reindex_rejects_index_based_on_different_commit(tmp_path):
    repo, store, _ = _setup_indexed_git_repo(
        tmp_path, {"tracked.py": "print('old')\n"}
    )
    (repo / "tracked.py").write_text("print('new')\n")
    _run_git(repo, "add", "tracked.py")
    _run_git(repo, "commit", "-m", "modify tracked file")
    previous_hash = store.get_file_hash("tracked.py")
    store.set_indexed_commit_sha("not-the-parent")

    with pytest.raises(ValueError, match="index base commit mismatch"):
        reindex(repo, parse_diff_output("M\ttracked.py"), store)

    reopened = VectorStore.open(repo / ".qatool" / "index")
    assert reopened.get_indexed_commit_sha() == "not-the-parent"
    assert reopened.get_file_hash("tracked.py") == previous_hash


def test_reindex_rejects_head_change_before_atomic_store_update(
    tmp_path, monkeypatch
):
    repo, store, _ = _setup_indexed_git_repo(
        tmp_path, {"tracked.py": "print('old')\n"}
    )
    store_path = repo / ".qatool" / "index" / "store.json"
    previous_data = store_path.read_text()
    (repo / "tracked.py").write_text("print('new')\n")
    _run_git(repo, "add", "tracked.py")
    _run_git(repo, "commit", "-m", "modify tracked file")
    target_sha = _run_git(repo, "rev-parse", "HEAD").stdout.strip()
    monkeypatch.setattr(
        "qatool.reindex.chunk_file",
        lambda full_path, rel_path: [
            Chunk(text=full_path.read_text(), metadata={"file": rel_path})
        ],
    )
    monkeypatch.setattr("qatool.reindex.embed_texts", lambda texts: [[2.0] for _ in texts])
    commit_states = iter([(target_sha, "base"), ("different-head", "base")])
    monkeypatch.setattr(
        "qatool.reindex.repository_commit_state", lambda repo_root: next(commit_states)
    )

    with pytest.raises(RuntimeError, match="HEAD changed during reindex"):
        reindex(repo, parse_diff_output("M\ttracked.py"), store)

    assert store_path.read_text() == previous_data


def _setup_hooked_repo(tmp_path: Path) -> tuple[Path, dict[str, str], Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init", "--initial-branch=master")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "user.email", "test@example.com")

    project_root = Path(__file__).resolve().parents[1]
    hooks_dir = repo / ".githooks"
    hooks_dir.mkdir()
    hook_path = hooks_dir / "post-commit"
    shutil.copyfile(project_root / ".githooks" / "post-commit", hook_path)
    hook_path.chmod(0o755)
    _run_git(repo, "config", "core.hooksPath", ".githooks")

    store_path = repo / ".qatool" / "index"
    store_path.mkdir(parents=True)
    (store_path / "store.json").write_text(json.dumps({"file_hashes": {}, "chunks": []}))

    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    argv_capture = capture_dir / "argv.json"
    diff_capture = capture_dir / "diff.txt"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    qatool_path = bin_dir / "qatool"
    qatool_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import os",
                "import shutil",
                "import sys",
                "from pathlib import Path",
                "",
                "argv_path = Path(os.environ['QATOOL_HOOK_ARGV_CAPTURE'])",
                "diff_path = Path(os.environ['QATOOL_HOOK_DIFF_CAPTURE'])",
                "argv_path.write_text(json.dumps(sys.argv[1:]))",
                "changed_files = Path(sys.argv[sys.argv.index('--changed-files') + 1])",
                "shutil.copyfile(changed_files, diff_path)",
                "",
            ]
        )
    )
    qatool_path.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["QATOOL_HOOK_ARGV_CAPTURE"] = str(argv_capture)
    env["QATOOL_HOOK_DIFF_CAPTURE"] = str(diff_capture)
    return repo, env, argv_capture, diff_capture


def test_post_commit_hook_invokes_reindex_with_repo_root_and_changed_files(tmp_path):
    repo, env, argv_capture, diff_capture = _setup_hooked_repo(tmp_path)

    (repo / "tracked.py").write_text("print('one')\n")
    (repo / "old_name.py").write_text("print('old')\n")
    (repo / "removed.txt").write_text("gone soon\n")
    _run_git(repo, "add", "tracked.py", "old_name.py", "removed.txt")
    _run_git(repo, "commit", "-m", "initial import", env=env)

    first_argv = json.loads(argv_capture.read_text())
    first_diff = diff_capture.read_text().splitlines()
    assert first_argv == [
        "reindex",
        "--repo",
        str(repo.resolve()),
        "--changed-files",
        first_argv[4],
    ]
    assert "A\ttracked.py" in first_diff
    assert "A\told_name.py" in first_diff
    assert "A\tremoved.txt" in first_diff

    (repo / "tracked.py").write_text("print('two')\n")
    _run_git(repo, "mv", "old_name.py", "new_name.py")
    _run_git(repo, "rm", "removed.txt")
    _run_git(repo, "add", "tracked.py")
    nested_dir = repo / "nested"
    nested_dir.mkdir()
    _run_git(repo, "commit", "-m", "update files", cwd=nested_dir, env=env)

    second_argv = json.loads(argv_capture.read_text())
    second_diff = diff_capture.read_text().splitlines()
    assert second_argv == [
        "reindex",
        "--repo",
        str(repo.resolve()),
        "--changed-files",
        second_argv[4],
    ]
    assert "M\ttracked.py" in second_diff
    assert "D\tremoved.txt" in second_diff
    assert any(line.startswith("R") and line.endswith("\told_name.py\tnew_name.py") for line in second_diff)


def test_post_commit_hook_uses_first_parent_merge_diff(tmp_path):
    repo, env, argv_capture, diff_capture = _setup_hooked_repo(tmp_path)

    (repo / "shared.txt").write_text("base\n")
    _run_git(repo, "add", "shared.txt")
    _run_git(repo, "commit", "-m", "base", env=env)

    _run_git(repo, "checkout", "-b", "feature")
    (repo / "feature_only.txt").write_text("feature\n")
    (repo / "shared.txt").write_text("base\nfeature\n")
    _run_git(repo, "add", "feature_only.txt", "shared.txt")
    _run_git(repo, "commit", "-m", "feature change", env=env)

    _run_git(repo, "checkout", "master")
    (repo / "main_only.txt").write_text("main\n")
    _run_git(repo, "add", "main_only.txt")
    _run_git(repo, "commit", "-m", "main change", env=env)

    _run_git(repo, "merge", "--no-commit", "--no-ff", "feature", env=env)
    _run_git(repo, "commit", "-m", "merge feature", env=env)

    argv = json.loads(argv_capture.read_text())
    diff_lines = diff_capture.read_text().splitlines()

    assert argv == [
        "reindex",
        "--repo",
        str(repo.resolve()),
        "--changed-files",
        argv[4],
    ]
    assert "A\tfeature_only.txt" in diff_lines
    assert "A\tmain_only.txt" not in diff_lines
    assert "M\tshared.txt" in diff_lines


def test_post_commit_hook_handles_resolved_merge_commits(tmp_path):
    repo, env, argv_capture, diff_capture = _setup_hooked_repo(tmp_path)

    (repo / "shared.txt").write_text("base\n")
    _run_git(repo, "add", "shared.txt")
    _run_git(repo, "commit", "-m", "base", env=env)

    _run_git(repo, "checkout", "-b", "feature")
    (repo / "shared.txt").write_text("base\nfeature\n")
    _run_git(repo, "add", "shared.txt")
    _run_git(repo, "commit", "-m", "feature change", env=env)

    _run_git(repo, "checkout", "master")
    (repo / "shared.txt").write_text("base\nmain\n")
    _run_git(repo, "add", "shared.txt")
    _run_git(repo, "commit", "-m", "main change", env=env)

    merge_result = _run_git(
        repo,
        "merge",
        "--no-commit",
        "--no-ff",
        "feature",
        env=env,
        check=False,
    )
    assert merge_result.returncode != 0

    (repo / "shared.txt").write_text("base\nmain\nfeature\n")
    _run_git(repo, "add", "shared.txt")
    _run_git(repo, "commit", "-m", "resolve merge", env=env)

    argv = json.loads(argv_capture.read_text())
    diff_lines = diff_capture.read_text().splitlines()

    assert argv == [
        "reindex",
        "--repo",
        str(repo.resolve()),
        "--changed-files",
        argv[4],
    ]
    assert diff_lines == ["M\tshared.txt"]


def test_post_commit_hook_invokes_reindex_for_empty_diff(tmp_path):
    repo, env, argv_capture, diff_capture = _setup_hooked_repo(tmp_path)

    (repo / "tracked.py").write_text("print('one')\n")
    _run_git(repo, "add", "tracked.py")
    _run_git(repo, "commit", "-m", "initial import", env=env)

    _run_git(repo, "commit", "--allow-empty", "-m", "empty commit", env=env)

    argv = json.loads(argv_capture.read_text())
    assert argv[:3] == ["reindex", "--repo", str(repo.resolve())]
    assert diff_capture.read_text() == ""


def test_reindex_main_requires_existing_local_index(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    changed_files = tmp_path / "changed.txt"
    changed_files.write_text("A\ttracked.py\n")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qatool-reindex",
            "--repo",
            str(repo),
            "--changed-files",
            str(changed_files),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        reindex_main()

    assert exc_info.value.code == 2
    error_output = capsys.readouterr().err
    assert "local index not found" in error_output
    assert str(repo / ".qatool" / "index" / "store.json") in error_output


def test_reindex_main_persists_commit_sha_for_empty_diff(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init", "--initial-branch=master")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "user.email", "test@example.com")
    (repo / "tracked.py").write_text("print('tracked')\n")
    _run_git(repo, "add", "tracked.py")
    _run_git(repo, "commit", "-m", "initial commit")
    commit_sha = _run_git(repo, "rev-parse", "HEAD").stdout.strip()
    changed_files = tmp_path / "changed.txt"
    changed_files.write_text("")
    store = VectorStore.open(repo / ".qatool" / "index")
    store.set_file_hash("tracked.py", file_hash(repo / "tracked.py"))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qatool-reindex",
            "--repo",
            str(repo),
            "--changed-files",
            str(changed_files),
        ],
    )

    reindex_main()

    assert store.get_indexed_commit_sha() == commit_sha
