"""Embedding provider abstractions and batched embedding calls."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from numbers import Real
from typing import Protocol


DEFAULT_EMBEDDING_BATCH_SIZE = 128
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BASE_DELAY_SECONDS = 0.5
DEFAULT_MOCK_DIMENSION = 32
DEFAULT_VOYAGE_ENDPOINT = "https://api.voyageai.com/v1/embeddings"
DEFAULT_VOYAGE_MODEL = "voyage-3-large"
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be generated."""


class EmbeddingRateLimitError(EmbeddingError):
    """Raised for rate-limit responses that may recover with retry."""


class EmbeddingTransientError(EmbeddingError):
    """Raised for transient provider failures that may recover with retry."""


class EmbeddingProvider(Protocol):
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text in the same order."""


@dataclass(frozen=True)
class EmbeddingConfig:
    provider_name: str
    model: str
    endpoint: str
    api_key: str | None
    batch_size: int
    max_retries: int
    retry_base_delay_seconds: float
    timeout_seconds: float
    mock_dimension: int


@dataclass(frozen=True)
class EmbeddingRuntimeConfig:
    batch_size: int
    max_retries: int
    retry_base_delay_seconds: float


def _read_int_env(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise EmbeddingError(
            f"invalid integer for {name}: {raw_value!r}"
        ) from exc
    if value < minimum:
        raise EmbeddingError(f"{name} must be >= {minimum}, got {value}")
    return value


def _read_float_env(name: str, default: float, minimum: float = 0.0) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise EmbeddingError(f"invalid float for {name}: {raw_value!r}") from exc
    if not math.isfinite(value):
        raise EmbeddingError(f"{name} must be finite, got {raw_value!r}")
    if value < minimum:
        raise EmbeddingError(f"{name} must be >= {minimum}, got {value}")
    return value


def load_embedding_config() -> EmbeddingConfig:
    provider_name = os.getenv("QATOOL_EMBEDDING_PROVIDER", "mock").strip().lower()
    return EmbeddingConfig(
        provider_name=provider_name,
        model=os.getenv("QATOOL_EMBEDDING_MODEL", DEFAULT_VOYAGE_MODEL),
        endpoint=os.getenv("QATOOL_EMBEDDING_ENDPOINT", DEFAULT_VOYAGE_ENDPOINT),
        api_key=os.getenv("QATOOL_EMBEDDING_API_KEY") or os.getenv("VOYAGE_API_KEY"),
        batch_size=_read_int_env("QATOOL_EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE),
        max_retries=_read_int_env("QATOOL_EMBEDDING_MAX_RETRIES", DEFAULT_MAX_RETRIES, minimum=0),
        retry_base_delay_seconds=_read_float_env(
            "QATOOL_EMBEDDING_RETRY_BASE_DELAY_SECONDS",
            DEFAULT_RETRY_BASE_DELAY_SECONDS,
            minimum=0.0,
        ),
        timeout_seconds=_read_float_env("QATOOL_EMBEDDING_TIMEOUT_SECONDS", 30.0, minimum=0.1),
        mock_dimension=_read_int_env("QATOOL_MOCK_EMBEDDING_DIMENSION", DEFAULT_MOCK_DIMENSION),
    )


def load_embedding_runtime_config() -> EmbeddingRuntimeConfig:
    return EmbeddingRuntimeConfig(
        batch_size=_read_int_env("QATOOL_EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE),
        max_retries=_read_int_env("QATOOL_EMBEDDING_MAX_RETRIES", DEFAULT_MAX_RETRIES, minimum=0),
        retry_base_delay_seconds=_read_float_env(
            "QATOOL_EMBEDDING_RETRY_BASE_DELAY_SECONDS",
            DEFAULT_RETRY_BASE_DELAY_SECONDS,
            minimum=0.0,
        ),
    )


class MockEmbeddingProvider:
    """Deterministic local provider used in tests and fallback development flows."""

    def __init__(self, dimension: int = DEFAULT_MOCK_DIMENSION) -> None:
        if dimension <= 0:
            raise EmbeddingError(f"mock dimension must be positive, got {dimension}")
        self.dimension = dimension

    def _embed_text(self, text: str) -> list[float]:
        tokens = TOKEN_RE.findall(text.lower()) or [text]
        vector = [0.0] * self.dimension

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            slot = digest[0] % self.dimension
            sign = -1.0 if digest[1] % 2 else 1.0
            weight = 1.0 + (digest[2] / 255.0)
            vector[slot] += sign * weight

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0.0:
            return vector
        return [value / magnitude for value in vector]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]


class VoyageEmbeddingProvider:
    """Voyage API-backed embedding provider."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_VOYAGE_MODEL,
        endpoint: str = DEFAULT_VOYAGE_ENDPOINT,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_key:
            raise EmbeddingError(
                "Voyage provider requires an API key via QATOOL_EMBEDDING_API_KEY or VOYAGE_API_KEY"
            )
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = json.dumps({"input": texts, "model": self.model}).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            method="POST",
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            status = getattr(exc, "code", None)
            detail = _read_http_error_body(exc)
            message = f"Voyage request failed with HTTP {status}{detail}"
            if status == 429:
                raise EmbeddingRateLimitError(message) from exc
            if status is not None and status >= 500:
                raise EmbeddingTransientError(message) from exc
            raise EmbeddingError(message) from exc
        except urllib.error.URLError as exc:
            transient_reasons = (
                TimeoutError,
                socket.timeout,
                ConnectionResetError,
                ConnectionAbortedError,
                BrokenPipeError,
            )
            if isinstance(exc.reason, transient_reasons):
                raise EmbeddingTransientError(f"Voyage request failed: {exc.reason}") from exc
            raise EmbeddingError(f"Voyage request failed: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise EmbeddingTransientError("Voyage request timed out") from exc

        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EmbeddingError("Voyage response was not valid JSON") from exc

        raw_data = decoded.get("data")
        if not isinstance(raw_data, list):
            raise EmbeddingError("Voyage response missing data array")

        vectors: list[list[float] | None] = [None] * len(texts)
        for fallback_index, item in enumerate(raw_data):
            if not isinstance(item, dict):
                raise EmbeddingError("Voyage response contains malformed data item")
            vector = item.get("embedding")
            if not isinstance(vector, list):
                raise EmbeddingError("Voyage response item missing embedding list")
            index = item.get("index", fallback_index)
            if not isinstance(index, int) or index < 0 or index >= len(vectors):
                raise EmbeddingError("Voyage response item has invalid index")
            if vectors[index] is not None:
                raise EmbeddingError(f"Voyage response contains duplicate index {index}")
            normalized: list[float] = []
            for value in vector:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise EmbeddingError("Voyage response item contains non-numeric embedding value")
                normalized.append(float(value))
            vectors[index] = normalized

        if any(vector is None for vector in vectors):
            raise EmbeddingError("Voyage response did not include all embedding vectors")
        return [vector for vector in vectors if vector is not None]


def _read_http_error_body(error: urllib.error.HTTPError) -> str:
    try:
        payload = error.read()
    except OSError:
        return ""
    if not payload:
        return ""
    try:
        decoded = payload.decode("utf-8").strip()
    except UnicodeDecodeError:
        return ""
    return f": {decoded}" if decoded else ""


def get_embedding_provider(config: EmbeddingConfig | None = None) -> EmbeddingProvider:
    config = config or load_embedding_config()
    if config.provider_name == "mock":
        return MockEmbeddingProvider(dimension=config.mock_dimension)
    if config.provider_name == "voyage":
        return VoyageEmbeddingProvider(
            api_key=config.api_key or "",
            model=config.model,
            endpoint=config.endpoint,
            timeout_seconds=config.timeout_seconds,
        )
    raise EmbeddingError(
        f"unsupported embedding provider {config.provider_name!r}; expected 'mock' or 'voyage'"
    )


def _embed_batch_with_retries(
    provider: EmbeddingProvider,
    batch: list[str],
    max_retries: int,
    retry_base_delay_seconds: float,
) -> list[list[float]]:
    attempts = max_retries + 1
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return provider.embed_batch(batch)
        except (EmbeddingRateLimitError, EmbeddingTransientError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = retry_base_delay_seconds * (2 ** (attempt - 1))
            if delay > 0:
                time.sleep(delay)
        except EmbeddingError:
            raise
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise EmbeddingError(f"embedding provider failed unexpectedly: {exc}") from exc

    message = f"embedding request failed after {attempts} attempt(s): {last_error}"
    if isinstance(last_error, (EmbeddingRateLimitError, EmbeddingTransientError)):
        raise type(last_error)(message) from last_error
    raise EmbeddingError(message) from last_error


def embed_texts(texts: list[str], provider: EmbeddingProvider | None = None) -> list[list[float]]:
    """Return one embedding vector per input text, in the same order."""
    if not texts:
        return []

    if provider is None:
        config = load_embedding_config()
        active_provider = get_embedding_provider(config)
        runtime_config = EmbeddingRuntimeConfig(
            batch_size=config.batch_size,
            max_retries=config.max_retries,
            retry_base_delay_seconds=config.retry_base_delay_seconds,
        )
    else:
        active_provider = provider
        runtime_config = load_embedding_runtime_config()
    vectors: list[list[float]] = []

    for start in range(0, len(texts), runtime_config.batch_size):
        batch = texts[start:start + runtime_config.batch_size]
        batch_vectors = _embed_batch_with_retries(
            active_provider,
            batch,
            max_retries=runtime_config.max_retries,
            retry_base_delay_seconds=runtime_config.retry_base_delay_seconds,
        )
        if len(batch_vectors) != len(batch):
            raise EmbeddingError(
                f"embedding count mismatch: expected {len(batch)}, got {len(batch_vectors)}"
            )
        vectors.extend(batch_vectors)

    if len(vectors) != len(texts):
        raise EmbeddingError(
            f"embedding count mismatch: expected {len(texts)}, got {len(vectors)}"
        )
    return vectors
