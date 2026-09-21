import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from qatool.reindex import main as reindex_main
from qatool.reindex import parse_diff_output


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


def _run_git(repo: Path, *args: str, cwd: Path | None = None, env: dict[str, str] | None = None):
    return subprocess.run(
        ["git", *args],
        cwd=cwd or repo,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def test_post_commit_hook_invokes_reindex_with_repo_root_and_changed_files(tmp_path):
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
