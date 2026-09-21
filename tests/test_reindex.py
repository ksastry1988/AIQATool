import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from qatool.chunking import Chunk
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


def _run_git(repo: Path, *args: str, cwd: Path | None = None, env: dict[str, str] | None = None):
    return subprocess.run(
        ["git", *args],
        cwd=cwd or repo,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def _setup_hooked_repo(tmp_path: Path) -> tuple[Path, dict[str, str], Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init")
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


def test_post_commit_hook_merges_changes_from_all_parents(tmp_path):
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
    assert "A\tmain_only.txt" in diff_lines
    assert "M\tshared.txt" in diff_lines


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
