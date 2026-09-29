import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

from qatool import qa
from qatool.cli import main as cli_main
from qatool.chunking import Chunk
from qatool.vectorstore import VectorStore


class FakeModelClient:
    def __init__(self, answer="Found it in the retrieval flow."):
        self.answer = answer
        self.question = None
        self.context = None

    def generate(self, question, context):
        self.question = question
        self.context = context
        return self.answer


def _indexed_repo(tmp_path):
    repo = tmp_path / "repo"
    store = VectorStore.open(repo / ".qatool" / "index")
    store.upsert(
        [
            Chunk(
                "def find_answer():\n    return 42\n",
                {
                    "file": "src/answer.py",
                    "start_line": 10,
                    "end_line": 11,
                    "symbol": "find_answer",
                    "language": "python",
                },
            ),
            Chunk(
                "class Other:\n    pass\n",
                {
                    "file": "src/other.py",
                    "start_line": 1,
                    "end_line": 2,
                    "symbol": "Other",
                    "language": "python",
                },
            ),
        ],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    return repo


def test_ask_question_retrieves_context_and_adds_source_citations(
    tmp_path, monkeypatch
):
    repo = _indexed_repo(tmp_path)
    client = FakeModelClient()
    monkeypatch.setattr(qa, "embed_texts", lambda texts: [[1.0, 0.0]])

    answer = qa.ask_question(repo, "Where is the answer?", top_k=1, model_client=client)

    assert client.question == "Where is the answer?"
    assert "Source: src/answer.py:L10-L11" in client.context
    assert "Symbol: find_answer" in client.context
    assert "def find_answer():" in client.context
    assert "src/other.py" not in client.context
    assert answer == (
        "Found it in the retrieval flow.\n\n"
        "Sources:\n- [src/answer.py:L10-L11]"
    )


def test_ask_question_reports_missing_index_without_creating_one(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        qa, "embed_texts", lambda texts: pytest.fail("must not embed without an index")
    )

    with pytest.raises(qa.AskError, match="local index not found"):
        qa.ask_question(repo, "question", model_client=FakeModelClient())

    assert not (repo / ".qatool").exists()


def test_ask_question_returns_no_results_without_calling_model(tmp_path, monkeypatch):
    repo = _indexed_repo(tmp_path)
    monkeypatch.setattr(qa, "embed_texts", lambda texts: [[1.0, 0.0]])

    class EmptyStore:
        STORE_FILENAME = VectorStore.STORE_FILENAME

        @staticmethod
        def open(path):
            return EmptyStore()

        @staticmethod
        def query(vector, top_k):
            return []

    monkeypatch.setattr(qa, "VectorStore", EmptyStore)
    assert qa.ask_question(repo, "question", model_client=None) == (
        "No relevant results found in the local index."
    )


def test_ask_question_validates_top_k_and_embedding_count(tmp_path, monkeypatch):
    repo = _indexed_repo(tmp_path)
    with pytest.raises(ValueError, match="top_k must be positive"):
        qa.ask_question(repo, "question", top_k=0)

    monkeypatch.setattr(qa, "embed_texts", lambda texts: [])
    with pytest.raises(qa.AskError, match="embedding count mismatch"):
        qa.ask_question(repo, "question", model_client=FakeModelClient())


def test_ask_question_rejects_empty_model_answer(tmp_path, monkeypatch):
    repo = _indexed_repo(tmp_path)
    monkeypatch.setattr(qa, "embed_texts", lambda texts: [[1.0, 0.0]])

    with pytest.raises(qa.AskError, match="empty answer"):
        qa.ask_question(repo, "question", model_client=FakeModelClient("  "))


def test_format_retrieved_chunks_handles_missing_metadata():
    assert qa.format_retrieved_chunks([{"text": "source text"}]) == (
        "[1] Source: unknown file\nsource text"
    )


def test_claude_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(qa.AskError, match="ANTHROPIC_API_KEY"):
        qa.ClaudeModelClient()


def test_claude_client_sends_context_and_extracts_text(monkeypatch):
    response = io.BytesIO(
        json.dumps(
            {
                "content": [
                    {"type": "text", "text": "First part."},
                    {"type": "tool_use", "id": "ignored"},
                    {"type": "text", "text": "Second part."},
                ]
            }
        ).encode()
    )
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return response

    monkeypatch.setattr(qa.urllib.request, "urlopen", fake_urlopen)
    client = qa.ClaudeModelClient(api_key="test-key", model="test-model")

    answer = client.generate("question", "retrieved context")

    sent_payload = json.loads(captured["request"].data)
    assert answer == "First part.\nSecond part."
    assert captured["request"].get_header("X-api-key") == "test-key"
    assert captured["timeout"] == 30
    assert sent_payload["model"] == "test-model"
    assert "retrieved context" in sent_payload["messages"][0]["content"]
    assert "only the supplied retrieved context" in sent_payload["system"]


@pytest.mark.parametrize(
    ("response_body", "error"),
    [
        (b"not-json", "not valid JSON"),
        (b'{"content": "invalid"}', "missing content"),
        (b'{"content": [{"type": "tool_use"}]}', "did not contain answer text"),
    ],
)
def test_claude_client_rejects_invalid_responses(monkeypatch, response_body, error):
    monkeypatch.setattr(
        qa.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(response_body),
    )

    with pytest.raises(qa.AskError, match=error):
        qa.ClaudeModelClient(api_key="test-key").generate("question", "context")


def test_claude_client_reports_http_errors(monkeypatch):
    def fail_request(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, 401, "unauthorized", {}, io.BytesIO(b"")
        )

    monkeypatch.setattr(qa.urllib.request, "urlopen", fail_request)
    with pytest.raises(qa.AskError, match="HTTP 401"):
        qa.ClaudeModelClient(api_key="test-key").generate("question", "context")


def test_claude_client_reports_connection_errors(monkeypatch):
    def fail_request(request, timeout):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(
        qa.urllib.request,
        "urlopen",
        fail_request,
    )

    with pytest.raises(qa.AskError, match="Claude request failed:"):
        qa.ClaudeModelClient(api_key="test-key").generate("question", "context")


def test_cli_ask_accepts_repo_and_top_k_options(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_ask(repo, question, top_k):
        captured.update(repo=repo, question=question, top_k=top_k)
        return "answer"

    monkeypatch.setattr(qa, "ask_question", fake_ask)
    monkeypatch.setattr(
        sys,
        "argv",
        ["qatool", "ask", "--repo", str(tmp_path), "--top-k", "3", "question"],
    )

    cli_main()

    assert captured == {
        "repo": Path(tmp_path),
        "question": "question",
        "top_k": 3,
    }
    assert capsys.readouterr().out == "answer\n"


def test_cli_ask_reports_missing_index(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["qatool", "ask", "--repo", str(tmp_path), "question"]
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_main()

    assert exc_info.value.code == 2
    assert "local index not found" in capsys.readouterr().err


def test_cli_ask_rejects_nonpositive_top_k(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["qatool", "ask", "--top-k", "0", "question"])

    with pytest.raises(SystemExit) as exc_info:
        cli_main()

    assert exc_info.value.code == 2
