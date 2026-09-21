import json
from urllib.error import HTTPError
from urllib.error import URLError

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


def test_embed_texts_with_explicit_provider_ignores_provider_selection_env(monkeypatch):
    monkeypatch.setenv("QATOOL_EMBEDDING_PROVIDER", "not-a-provider")
    provider = RecordingProvider()
    vectors = embed_texts(["alpha"], provider=provider)
    assert vectors == [[1.0]]


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


def test_embed_texts_preserves_rate_limit_error_type_after_retry_exhaustion(monkeypatch):
    monkeypatch.setenv("QATOOL_EMBEDDING_MAX_RETRIES", "1")

    class RateLimitedProvider:
        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            raise EmbeddingRateLimitError("too many requests")

    with pytest.raises(EmbeddingRateLimitError) as exc_info:
        embed_texts(["alpha"], provider=RateLimitedProvider())
    assert "failed after 2 attempt(s)" in str(exc_info.value)


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


def test_get_embedding_provider_applies_mock_dimension_env(monkeypatch):
    monkeypatch.setenv("QATOOL_EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("QATOOL_MOCK_EMBEDDING_DIMENSION", "7")
    provider = get_embedding_provider()
    vectors = provider.embed_batch(["alpha"])
    assert len(vectors[0]) == 7


def test_get_embedding_provider_applies_voyage_env_settings(monkeypatch):
    monkeypatch.setenv("QATOOL_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("QATOOL_EMBEDDING_API_KEY", "key-from-env")
    monkeypatch.setenv("QATOOL_EMBEDDING_MODEL", "voyage-3-lite")
    monkeypatch.setenv("QATOOL_EMBEDDING_ENDPOINT", "https://example.invalid/embed")
    monkeypatch.setenv("QATOOL_EMBEDDING_TIMEOUT_SECONDS", "12.5")

    provider = get_embedding_provider()
    assert isinstance(provider, VoyageEmbeddingProvider)
    assert provider.api_key == "key-from-env"
    assert provider.model == "voyage-3-lite"
    assert provider.endpoint == "https://example.invalid/embed"
    assert provider.timeout_seconds == 12.5


def test_get_embedding_provider_rejects_non_finite_timeout_env(monkeypatch):
    monkeypatch.setenv("QATOOL_EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("QATOOL_EMBEDDING_TIMEOUT_SECONDS", "nan")
    with pytest.raises(EmbeddingError) as exc_info:
        get_embedding_provider()
    assert "must be finite" in str(exc_info.value)


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
        auth_header = dict(request.header_items()).get("Authorization")
        assert auth_header is not None
        assert auth_header.startswith("Bearer ")
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


def test_voyage_provider_empty_batch_returns_without_request(monkeypatch):
    provider = VoyageEmbeddingProvider(api_key="test-key")

    def fail_if_called(request, timeout):
        raise AssertionError("urlopen should not be called for empty input")

    monkeypatch.setattr("qatool.embeddings.urllib.request.urlopen", fail_if_called)
    assert provider.embed_batch([]) == []


def test_voyage_provider_classifies_timeouts_as_transient(monkeypatch):
    provider = VoyageEmbeddingProvider(api_key="test-key")

    monkeypatch.setattr(
        "qatool.embeddings.urllib.request.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(TimeoutError("timed out")),
    )
    with pytest.raises(EmbeddingTransientError):
        provider.embed_batch(["alpha"])


def test_voyage_provider_classifies_urlerror_timeouts_as_transient(monkeypatch):
    provider = VoyageEmbeddingProvider(api_key="test-key")

    def raise_urlerror_timeout(request, timeout):
        raise URLError(TimeoutError("timed out"))

    monkeypatch.setattr("qatool.embeddings.urllib.request.urlopen", raise_urlerror_timeout)
    with pytest.raises(EmbeddingTransientError):
        provider.embed_batch(["alpha"])


def test_voyage_provider_classifies_non_transient_urlerror_as_fatal(monkeypatch):
    provider = VoyageEmbeddingProvider(api_key="test-key")

    def raise_urlerror(request, timeout):
        raise URLError("unknown url type: ftp")

    monkeypatch.setattr("qatool.embeddings.urllib.request.urlopen", raise_urlerror)
    with pytest.raises(EmbeddingError) as exc_info:
        provider.embed_batch(["alpha"])
    assert "unknown url type: ftp" in str(exc_info.value)


def test_voyage_provider_rejects_duplicate_indexes(monkeypatch):
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
                        {"index": 0, "embedding": [1.0]},
                        {"index": 0, "embedding": [2.0]},
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setattr("qatool.embeddings.urllib.request.urlopen", lambda request, timeout: Response())
    with pytest.raises(EmbeddingError) as exc_info:
        provider.embed_batch(["alpha", "beta"])
    assert "duplicate index" in str(exc_info.value)


def test_voyage_provider_rejects_non_numeric_embedding_values(monkeypatch):
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
                        {"index": 0, "embedding": [1.0]},
                        {"index": 1, "embedding": [True]},
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setattr("qatool.embeddings.urllib.request.urlopen", lambda request, timeout: Response())
    with pytest.raises(EmbeddingError) as exc_info:
        provider.embed_batch(["alpha", "beta"])
    assert "non-numeric embedding value" in str(exc_info.value)
