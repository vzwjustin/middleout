"""Embedding clients for the L2 semantic cache.

Two implementations ship with the proxy:

- ``HashEmbedder`` — stdlib-only, deterministic, fast. NOT a real semantic
  embedder. It hashes shingles of the normalized payload into a high-dim float
  vector. Tests use it; in production it acts as a "warmup" embedder that gives
  L2 hits on byte-identical or near-identical (whitespace-only) payloads — i.e.
  ~ what L1 already does. Useful as a default when the operator has not yet
  wired a real embedding API.

- ``OpenAIEmbeddingClient`` — adapter for OpenAI's embeddings endpoint. Behind
  an import guard so the ``openai`` package is optional. Uses the configured
  model (default ``text-embedding-3-large``, 3072-dim — operator-overridable).

Both implementations expose the ``EmbeddingClient`` protocol from
``cache.l2``. Operators with their own embedding source (Anthropic future
endpoint, local sentence-transformer, Cohere, Voyage, etc.) can supply any
object that matches the protocol — no inheritance required.

Determinism + safety
--------------------
- Embedders are NEVER fed raw user content directly. The integration layer
  passes a **normalized** payload string (the same canonical JSON used for the
  L1 key). This keeps the cache deterministic across whitespace/key-order
  noise.

- Embedding API calls happen synchronously on the request hot path. The
  ``OpenAIEmbeddingClient`` enforces a hard timeout so a slow embeddings API
  cannot stall a request indefinitely.

- The embedder is shared across requests; vectors are computed per request, so
  thread/asyncio safety is the underlying library's responsibility (OpenAI's
  Python client is thread-safe).
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)


# Default dim for the hash embedder. Chosen to match the spec's
# `text-embedding-3-large` (3072) — operators swapping in OpenAI without
# changing other config see consistent dimensionality.
_DEFAULT_HASH_DIM = 3072


class HashEmbedder:
    """Deterministic stdlib-only pseudo-embedder.

    Produces an L2-normalized vector by hashing word shingles into bucket
    indices and accumulating ±1 contributions per shingle. Two strings that
    differ in only whitespace or key order map to *similar* vectors because
    canonical JSON normalization (done by the integration layer before
    embedding) collapses those differences. Two strings with substantively
    different content map to *different* vectors.

    NOT a substitute for a real semantic embedder. Use it for:
    - tests (deterministic, no network)
    - flag-on-with-no-OpenAI-key dev environments
    - benchmark baselines

    Switch to ``OpenAIEmbeddingClient`` (or another protocol implementation)
    for production semantic recall.
    """

    def __init__(
        self,
        *,
        dim: int = _DEFAULT_HASH_DIM,
        shingle_chars: int = 4,
    ) -> None:
        if dim < 16:
            raise ValueError(f"dim must be at least 16, got {dim}")
        if shingle_chars < 1:
            raise ValueError(f"shingle_chars must be >= 1, got {shingle_chars}")
        self.dim = dim
        self.shingle_chars = shingle_chars

    def embed(self, text: str) -> list[float]:
        """Return a unit-length pseudo-embedding for `text`.

        Algorithm: walk over character-level shingles of width `shingle_chars`,
        hash each shingle with blake2b, derive a (bucket, sign) from the hash,
        accumulate ±1 into the bucket, then L2-normalize.
        """
        if not isinstance(text, str):
            text = repr(text)

        vec = [0.0] * self.dim
        # Pad short inputs so very tiny payloads still produce a meaningful
        # vector (rather than the zero vector which would compare equal to
        # every other zero vector under cosine).
        padded = text if len(text) >= self.shingle_chars else (
            text + "\x00" * (self.shingle_chars - len(text))
        )
        for i in range(len(padded) - self.shingle_chars + 1):
            shingle = padded[i : i + self.shingle_chars]
            digest = hashlib.blake2b(
                shingle.encode("utf-8", errors="replace"),
                digest_size=8,
            ).digest()
            # Use the first 4 bytes for the bucket index, the next byte for sign.
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        inv = 1.0 / norm
        return [v * inv for v in vec]


class OpenAIEmbeddingClient:
    """Adapter for the OpenAI Embeddings API.

    Import-guarded — the ``openai`` package is an optional dep. Construct
    raises ``ImportError`` with a helpful message if it's missing. Once
    constructed, ``embed`` calls the API with a per-call timeout.

    Authentication: reads ``OPENAI_API_KEY`` from env by default. Pass
    ``api_key=`` to override (e.g., from a TOML config).
    """

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-large",
        api_key: str | None = None,
        timeout_s: float = 10.0,
        dim: int = 3072,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "OpenAIEmbeddingClient requires the `openai` package. "
                "Install with: `pip install openai`."
            ) from e
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is not set and no `api_key=` was supplied."
            )
        client_kwargs: dict[str, Any] = {"api_key": key, "timeout": timeout_s}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)
        self.model = model
        self.dim = dim
        self.timeout_s = timeout_s

    def embed(self, text: str) -> list[float]:
        try:
            resp = self._client.embeddings.create(
                model=self.model,
                input=text,
                # Newer OpenAI embeddings accept a `dimensions` arg to truncate
                # the vector to a smaller size. Passing matches what the
                # configured Qdrant collection expects.
                dimensions=self.dim,
            )
            data = resp.data[0]
            embedding = list(data.embedding)
        except TypeError:
            # Older API surfaces that don't accept `dimensions=`.
            resp = self._client.embeddings.create(model=self.model, input=text)
            embedding = list(resp.data[0].embedding)
        if len(embedding) != self.dim:
            logger.warning(
                "OpenAI returned %d-dim embedding; expected %d. "
                "Update settings.l2_embedding_dim or the model.",
                len(embedding),
                self.dim,
            )
        return embedding


__all__ = ["HashEmbedder", "OpenAIEmbeddingClient"]
