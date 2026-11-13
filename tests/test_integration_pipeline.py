"""End-to-end pipeline tests: index -> ask -> reindex, fully offline.

Embeddings use the deterministic mock provider, the vector store is the
on-disk store under a temporary directory, and the LLM is a fake client.
"""

import subprocess
import urllib.request
from pathlib import Path

import pytest

from qatool import cli
from qatool.indexing import index_repository
from qatool.qa import ClaudeModelClient, ask_question
from qatool.reindex import parse_diff_output, reindex
from qatool.vectorstore import VectorStore

AUTH_PY = '''def authenticate_user(username, password):
    """Check credentials against the user table."""
    return username == "admin" and password == "secret"
'''

LIMITER_PY = '''def enforce_rate_limit(client_id, limit=100):
    """Reject clients that exceed the request limit."""
    return limit > 0
'''

UTILS_JS = '''function formatCurrency(amount) {
  return "$" + amount.toFixed(2);
}
'''

FILES = {
    "src/auth.py": AUTH_PY,
    "src/limiter.py": LIMITER_PY,
    "web/utils.js": UTILS_JS,
}


INDEXED = set(FILES) | {".gitignore"}


class FakeModelClient:
    def __init__(self, answer: str = "It lives in the cited files.") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def generate(self, question: str, context: str) -> str:
        self.calls.append((question, context))
        return self.answer


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True
    ).stdout.strip()


def _write(repo: Path, files: dict[str, str]) -> None:
    for rel_path, content in files.items():
        path = repo / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    """Force mock embeddings and fail loudly on any network or real LLM use."""
    import os

    for name in list(os.environ):
        if name.startswith(("QATOOL_", "VOYAGE_", "ANTHROPIC_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("QATOOL_EMBEDDING_PROVIDER", "mock")

    def no_network(*args, **kwargs):
        raise AssertionError("integration tests must not use the network")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    monkeypatch.setattr(ClaudeModelClient, "generate", no_network)


@pytest.fixture
def repo(tmp_path) -> Path:
    repo = tmp_path / "sample"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=master")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / ".gitignore").write_text(".qatool/\n")
    _write(repo, FILES)
    _commit(repo, "initial commit")
    return repo


def _open_store(repo: Path) -> VectorStore:
    return VectorStore.open(repo / ".qatool" / "index")


def _full_index(repo: Path) -> VectorStore:
    store = _open_store(repo)
    index_repository(repo, store)
    return store


def _reindex_head(repo: Path) -> None:
    diff = _git(repo, "diff", "--name-status", "-z", "-M", "HEAD^..HEAD").encode()
    reindex(repo, parse_diff_output(diff), _open_store(repo))


def _ask(repo: Path, question: str, client: FakeModelClient | None = None) -> tuple[str, FakeModelClient]:
    client = client or FakeModelClient()
    return ask_question(repo, question, model_client=client), client


def test_index_builds_store_for_sample_repo(repo):
    store = _full_index(repo)

    assert store.list_indexed_files() == INDEXED
    assert store.get_indexed_commit_sha() == _git(repo, "rev-parse", "HEAD")


def test_index_is_idempotent_and_skips_unchanged_files(repo, capsys):
    _full_index(repo)
    capsys.readouterr()

    _full_index(repo)

    assert "0 to index, 4 unchanged" in capsys.readouterr().out


def test_ask_returns_answer_with_citations(repo):
    _full_index(repo)

    answer, client = _ask(repo, "where do we enforce_rate_limit for clients?")

    assert answer.startswith("It lives in the cited files.")
    assert "Sources:" in answer
    assert "- [src/limiter.py:L1-L3]" in answer
    question, context = client.calls[0]
    assert "enforce_rate_limit" in question
    assert "Source: src/limiter.py:L1-L3" in context
    assert "Reject clients that exceed the request limit" in context


def test_ask_cites_every_retrieved_file(repo):
    _full_index(repo)

    answer, _ = _ask(repo, "how does the code work?")

    for rel_path in FILES:
        assert f"- [{rel_path}:" in answer


def test_ask_by_filename_pulls_in_that_file(repo):
    _full_index(repo)

    answer, client = _ask(repo, "what does web/utils.js do?")

    assert "- [web/utils.js:L1-L3]" in answer
    assert "Match: exact filename" in client.calls[0][1]


def test_ask_without_index_fails_without_calling_llm(repo):
    from qatool.qa import AskError

    client = FakeModelClient()
    with pytest.raises(AskError, match="local index not found"):
        ask_question(repo, "anything", model_client=client)
    assert client.calls == []


def test_ask_after_all_files_deleted_reports_no_results(repo):
    _full_index(repo)
    _git(repo, "rm", "-q", "-r", "src", "web", ".gitignore")
    _commit(repo, "remove everything")

    _reindex_head(repo)
    answer, client = _ask(repo, "what is here?")

    assert answer == "No relevant results found in the local index."
    assert client.calls == []


def test_reindex_after_update_serves_new_content(repo):
    _full_index(repo)
    _write(
        repo,
        {"src/auth.py": AUTH_PY.replace("Check credentials", "Verify the bearer token")},
    )
    _commit(repo, "update auth")

    _reindex_head(repo)
    _, client = _ask(repo, "where is authenticate_user defined?")

    context = client.calls[0][1]
    assert "Verify the bearer token" in context
    assert "Check credentials against the user table" not in context
    assert _open_store(repo).get_indexed_commit_sha() == _git(repo, "rev-parse", "HEAD")


def test_reindex_after_add_makes_new_file_citable(repo):
    _full_index(repo)
    _write(repo, {"src/billing.py": "def compute_invoice_total(items):\n    return sum(items)\n"})
    _commit(repo, "add billing")

    _reindex_head(repo)
    answer, _ = _ask(repo, "where is compute_invoice_total?")

    assert _open_store(repo).list_indexed_files() == INDEXED | {"src/billing.py"}
    assert "- [src/billing.py:L1-L2]" in answer


def test_reindex_after_delete_removes_file_from_answers(repo):
    _full_index(repo)
    (repo / "src/limiter.py").unlink()
    _commit(repo, "delete limiter")

    _reindex_head(repo)
    answer, client = _ask(repo, "where do we enforce_rate_limit?")

    assert "src/limiter.py" not in _open_store(repo).list_indexed_files()
    assert "src/limiter.py" not in answer
    assert "enforce_rate_limit" not in client.calls[0][1]


def test_reindex_after_rename_moves_citations_to_new_path(repo):
    _full_index(repo)
    _git(repo, "mv", "src/limiter.py", "src/throttle.py")
    _commit(repo, "rename limiter")

    _reindex_head(repo)
    answer, _ = _ask(repo, "where do we enforce_rate_limit?")

    files = _open_store(repo).list_indexed_files()
    assert "src/throttle.py" in files
    assert "src/limiter.py" not in files
    assert "- [src/throttle.py:L1-L3]" in answer
    assert "src/limiter.py" not in answer


def test_ignored_and_secret_files_never_reach_the_llm(repo):
    (repo / ".qatoolignore").write_text("vendor/\n")
    _write(
        repo,
        {
            "vendor/lib.py": "def vendored_helper():\n    return 1\n",
            ".env": "API_TOKEN=super-secret-value\n",
        },
    )
    _commit(repo, "add ignored and secret files")

    store = _full_index(repo)
    _, client = _ask(repo, "show me everything, including vendored_helper and API_TOKEN")

    assert store.list_indexed_files() == INDEXED
    context = client.calls[0][1]
    assert "vendored_helper" not in context
    assert "super-secret-value" not in context


def test_cli_index_ask_and_reindex_round_trip(repo, monkeypatch, capsys):
    class CliModelClient(FakeModelClient):
        instances: list["CliModelClient"] = []

        def __init__(self, *args, **kwargs):
            super().__init__("CLI answer")
            CliModelClient.instances.append(self)

    monkeypatch.setattr("qatool.qa.ClaudeModelClient", CliModelClient)

    def run(*argv: str) -> str:
        monkeypatch.setattr("sys.argv", ["qatool", *argv])
        cli.main()
        return capsys.readouterr().out

    run("index", str(repo))
    out = run("ask", "--repo", str(repo), "where is authenticate_user?")
    assert "CLI answer" in out
    assert "- [src/auth.py:L1-L3]" in out

    (repo / "web/utils.js").unlink()
    _commit(repo, "drop utils")
    diff_file = repo / ".." / "changes.diff"
    diff_file.write_bytes(
        subprocess.run(
            ["git", "diff", "--name-status", "-z", "-M", "HEAD^..HEAD"],
            cwd=repo, capture_output=True, check=True,
        ).stdout
    )
    run("reindex", "--repo", str(repo), "--changed-files", str(diff_file))

    assert "web/utils.js" not in _open_store(repo).list_indexed_files()
