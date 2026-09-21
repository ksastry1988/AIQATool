import json
from urllib.error import HTTPError

import pytest

from qatool.embeddings import (
    EmbeddingError,
    EmbeddingRateLimitError,
    EmbeddingTransientError,
    VoyageEmbeddingProvider,
    embed_texts,
    get_embedding_provider,
)


class RecordingProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(index)] for index, _ in enumerate(texts, start=1)]


def test_embed_texts_returns_one_vector_per_input():
    vectors = embed_texts(["alpha", "beta"])
    assert len(vectors) == 2


def test_embed_texts_batches_large_inputs(monkeypatch):
    monkeypatch.setenv("QATOOL_EMBEDDING_BATCH_SIZE", "2")
    provider = RecordingProvider()

    vectors = embed_texts(["a", "b", "c", "d", "e"], provider=provider)

    assert provider.calls == [["a", "b"], ["c", "d"], ["e"]]
    assert len(vectors) == 5


def test_embed_texts_retries_rate_limit_failures(monkeypatch):
    monkeypatch.setenv("QATOOL_EMBEDDING_MAX_RETRIES", "2")
    monkeypatch.setenv("QATOOL_EMBEDDING_RETRY_BASE_DELAY_SECONDS", "0.01")
    delays: list[float] = []
    monkeypatch.setattr("qatool.embeddings.time.sleep", delays.append)

    attempts = {"count": 0}

    class FlakyProvider:
        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise EmbeddingRateLimitError("too many requests")
            return [[1.0] for _ in texts]

    vectors = embed_texts(["alpha"], provider=FlakyProvider())
    assert vectors == [[1.0]]
    assert attempts["count"] == 3
    assert delays == [0.01, 0.02]


def test_embed_texts_reports_clear_errors_for_exhausted_retries(monkeypatch):
    monkeypatch.setenv("QATOOL_EMBEDDING_MAX_RETRIES", "1")

    class BrokenProvider:
        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            raise EmbeddingTransientError("gateway timeout")

    with pytest.raises(EmbeddingError) as exc_info:
        embed_texts(["alpha"], provider=BrokenProvider())

    message = str(exc_info.value)
    assert "failed after 2 attempt(s)" in message
    assert "gateway timeout" in message


def test_embed_texts_validates_vector_count(monkeypatch):
    monkeypatch.setenv("QATOOL_EMBEDDING_BATCH_SIZE", "10")

    class InvalidProvider:
        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] for _ in texts[:-1]]

    with pytest.raises(EmbeddingError) as exc_info:
        embed_texts(["alpha", "beta"], provider=InvalidProvider())

    assert "embedding count mismatch: expected 2, got 1" in str(exc_info.value)


def test_get_embedding_provider_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("QATOOL_EMBEDDING_PROVIDER", "unknown")
    with pytest.raises(EmbeddingError) as exc_info:
        get_embedding_provider()
    assert "unsupported embedding provider" in str(exc_info.value)


def test_get_embedding_provider_requires_voyage_api_key(monkeypatch):
    monkeypatch.setenv("QATOOL_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.delenv("QATOOL_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    with pytest.raises(EmbeddingError) as exc_info:
        get_embedding_provider()
    assert "requires an API key" in str(exc_info.value)


def test_voyage_provider_parses_embeddings_in_original_order(monkeypatch):
    provider = VoyageEmbeddingProvider(api_key="test-key")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "data": [
                        {"index": 1, "embedding": [2.0]},
                        {"index": 0, "embedding": [1.0]},
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        assert request.full_url == "https://api.voyageai.com/v1/embeddings"
        assert timeout == 30.0
        assert request.get_header("Authorization") == "******"
        return Response()

    monkeypatch.setattr("qatool.embeddings.urllib.request.urlopen", fake_urlopen)
    vectors = provider.embed_batch(["alpha", "beta"])
    assert vectors == [[1.0], [2.0]]


def test_voyage_provider_classifies_http_429_as_rate_limit(monkeypatch):
    provider = VoyageEmbeddingProvider(api_key="test-key")

    def fake_urlopen(request, timeout):
        raise HTTPError(
            url=request.full_url,
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("qatool.embeddings.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(EmbeddingRateLimitError):
        provider.embed_batch(["alpha"])
