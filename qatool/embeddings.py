"""
Batched embedding calls.

TODO: wire up to Voyage AI (recommended pairing with Claude) or your
provider of choice. Keep this the only module that knows which
embedding API is in use, so swapping providers later stays cheap.
"""

from __future__ import annotations

import hashlib
import math
import re


EMBEDDING_DIMENSION = 32
BATCH_SIZE = 128
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _embed_text(text: str) -> list[float]:
    tokens = TOKEN_RE.findall(text.lower()) or [text]
    vector = [0.0] * EMBEDDING_DIMENSION

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        slot = digest[0] % EMBEDDING_DIMENSION
        sign = -1.0 if digest[1] % 2 else 1.0
        weight = 1.0 + (digest[2] / 255.0)
        vector[slot] += sign * weight

    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0.0:
        return vector
    return [value / magnitude for value in vector]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Return one embedding vector per input text, same order.

    Batch requests where the provider allows it — most embedding APIs
    charge per call as well as per token, so batching matters at scale.
    """
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        vectors.extend(_embed_text(text) for text in batch)
    return vectors
