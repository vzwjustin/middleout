"""L2 semantic response cache stub (Phase 2b).

The L1 cache hits when two requests serialize to the same bytes; L2 hits when
two requests *mean* the same thing. The mechanism is: embed each request,
ANN-lookup against a vector store (Qdrant by default), check similarity
against a threshold, then optionally run a cheap verification step (e.g. an
LLM-judge or token-level overlap) before serving.

This file scaffolds the public surface. The actual embedding + Qdrant client
is intentionally not implemented yet — getting the interface right is the
hard part; the body shape can be filled in once we wire embeddings.

Interface
---------
``L2Cache`` exposes the same `get(key) -> CachedResponse | None` and
`put(key, response)` contract as :class:`L1Cache`, with an extra
`get_similar(payload, *, threshold)` for the semantic lookup. The integration
layer calls L1 first; on L1 miss, calls L2; on L2 hit, optionally verifies and
serves; on either miss, talks to upstream and writes both.

Embeddings
----------
The default embedder is the configured Qdrant collection's expected dimension
(1536 for ``text-embedding-3-large`` per the user spec). The proxy itself
does NOT call the embedding API — that would defeat the cache's whole point.
Instead, the operator's preferred embedding source is plugged in via the
:class:`EmbeddingClient` protocol.

Why a stub
----------
We have the L1 cache working end-to-end, with tests and a 5xx-safe wire path.
Phase 2b adds significant moving parts (embedding model availability, network
to Qdrant, verification semantics) and deserves its own design pass before
shipping. The stub keeps the public types stable so the integration layer can
already speak L2 in flag-off mode, even though the lookup always misses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from .l1 import CachedResponse


logger = logging.getLogger(__name__)


class L2NotConfigured(RuntimeError):
    """Raised when the operator enabled L2 without supplying an embedding client."""


@dataclass(frozen=True)
class SemanticHit:
    """One ANN result. ``similarity`` is in [0, 1]; higher = closer."""

    similarity: float
    response: CachedResponse
    point_id: str
    metadata: dict[str, Any]


class EmbeddingClient(Protocol):
    """Caller-supplied embedding source.

    The proxy does not bake in a specific model. Implementations may call
    OpenAI's embeddings API, a local sentence-transformer, or Anthropic's
    future embeddings endpoint. ``dim`` must match the configured Qdrant
    collection dimension.
    """

    dim: int

    def embed(self, text: str) -> list[float]:
        """Return a single embedding vector for `text`."""


class VectorStore(Protocol):
    """Caller-supplied ANN backend.

    The default implementation we'll ship will wrap qdrant-client. The
    interface is narrow on purpose so other backends (faiss, chroma) can drop
    in later.
    """

    def upsert(
        self,
        point_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Insert or replace a point."""

    def search(
        self,
        vector: list[float],
        *,
        top_k: int = 5,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Return up to `top_k` `(point_id, similarity, payload)` results."""

    def delete(self, point_id: str) -> None:
        """Remove a point by id."""


class L2Cache:
    """Semantic response cache. **Stub** — every lookup misses for now.

    The integration layer wires `L2Cache` behind a feature flag. With the flag
    off (default), no embedding calls and no Qdrant traffic happen; with it on
    but no `embedding_client` supplied, :class:`L2NotConfigured` is raised at
    construction so the operator sees the misconfiguration immediately.
    """

    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient | None = None,
        vector_store: VectorStore | None = None,
        similarity_threshold: float = 0.97,
        enabled: bool = False,
    ) -> None:
        if enabled and (embedding_client is None or vector_store is None):
            raise L2NotConfigured(
                "L2 semantic cache is enabled but no embedding_client or "
                "vector_store was provided. Either disable L2 or wire both."
            )
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.similarity_threshold = float(similarity_threshold)
        self.enabled = bool(enabled)
        self.lookups = 0
        self.hits = 0

    # -- public API -------------------------------------------------------

    def get_similar(
        self,
        normalized_payload_text: str,
        *,
        threshold: float | None = None,
    ) -> SemanticHit | None:
        """Embed `normalized_payload_text` and ANN-lookup. None on miss/disabled.

        `normalized_payload_text` is the canonical JSON form used for the L1
        key — embedding the same string keeps the two layers' notion of
        "identical" aligned.
        """
        if not self.enabled or self.embedding_client is None or self.vector_store is None:
            return None
        try:
            self.lookups += 1
            vec = self.embedding_client.embed(normalized_payload_text)
            results = self.vector_store.search(vec, top_k=1)
        except Exception as e:
            logger.warning("L2 lookup failed: %s: %s", type(e).__name__, e)
            return None
        if not results:
            return None
        point_id, similarity, metadata = results[0]
        eff_threshold = self.similarity_threshold if threshold is None else float(threshold)
        if similarity < eff_threshold:
            return None
        response = _metadata_to_response(metadata)
        if response is None:
            return None
        self.hits += 1
        return SemanticHit(
            similarity=similarity,
            response=response,
            point_id=point_id,
            metadata=metadata,
        )

    def put_similar(
        self,
        normalized_payload_text: str,
        response: CachedResponse,
        *,
        point_id: str,
    ) -> None:
        """Embed and store. No-op when L2 is disabled."""
        if not self.enabled or self.embedding_client is None or self.vector_store is None:
            return
        try:
            vec = self.embedding_client.embed(normalized_payload_text)
            self.vector_store.upsert(
                point_id=point_id,
                vector=vec,
                payload=_response_to_metadata(response),
            )
        except Exception as e:
            logger.warning("L2 put failed: %s: %s", type(e).__name__, e)

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "lookups": self.lookups,
            "hits": self.hits,
            "threshold": self.similarity_threshold,
            "embedding_dim": getattr(self.embedding_client, "dim", None),
        }


def _response_to_metadata(response: CachedResponse) -> dict[str, Any]:
    """Serialize a CachedResponse to a Qdrant payload."""
    import base64

    return {
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "body_b64": base64.b64encode(response.body).decode("ascii"),
        "media_type": response.media_type,
        "inserted_at": response.inserted_at,
        "hit_count": response.hit_count,
    }


def _metadata_to_response(metadata: dict[str, Any]) -> CachedResponse | None:
    """Deserialize Qdrant payload back to a CachedResponse."""
    import base64

    try:
        body_b64 = metadata.get("body_b64")
        body = base64.b64decode(body_b64) if isinstance(body_b64, str) else b""
        return CachedResponse(
            status_code=int(metadata.get("status_code", 200)),
            headers=dict(metadata.get("headers", {})),
            body=body,
            media_type=metadata.get("media_type"),
            inserted_at=float(metadata.get("inserted_at", 0.0)),
            hit_count=int(metadata.get("hit_count", 0)),
        )
    except (TypeError, ValueError, KeyError):
        return None


__all__ = [
    "EmbeddingClient",
    "L2Cache",
    "L2NotConfigured",
    "SemanticHit",
    "VectorStore",
]
