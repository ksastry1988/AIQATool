"""
Batched embedding calls.

TODO: wire up to Voyage AI (recommended pairing with Claude) or your
provider of choice. Keep this the only module that knows which
embedding API is in use, so swapping providers later stays cheap.
"""

from __future__ import annotations


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Return one embedding vector per input text, same order.

    Batch requests where the provider allows it — most embedding APIs
    charge per call as well as per token, so batching matters at scale.
    """
    raise NotImplementedError("Wire up your embedding provider here.")
