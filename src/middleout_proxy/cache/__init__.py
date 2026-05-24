"""Two-tier response cache.

Phase 2a: L1 exact-match (SHA256 of normalized request payload → response).
Phase 2b: L2 semantic similarity via embeddings + ANN (Qdrant). Stubbed —
the public types are stable so the integration layer can speak L2 in
flag-off mode, but the lookup always misses until embeddings ship.

Public API
----------
- `cache_key(payload)` — canonical sha256 hex of a payload after normalization.
- `L1Cache` — SQLite-backed store with bounded entry count + LRU eviction.
- `L2Cache` — semantic ANN cache stub.
- `CachedResponse` — dataclass carrying status, headers, body, and metadata.

The cache lives behind a feature flag (`BRAIN_L1_CACHE_ENABLED`) and is off by
default. Streaming requests are NOT cached in this phase — SSE chunk boundary
preservation is non-trivial and a separate follow-up.
"""

from .l1 import CachedResponse, L1Cache
from .l2 import EmbeddingClient, L2Cache, L2NotConfigured, SemanticHit, VectorStore
from .normalize import cache_key, normalize_payload

__all__ = [
    "CachedResponse",
    "EmbeddingClient",
    "L1Cache",
    "L2Cache",
    "L2NotConfigured",
    "SemanticHit",
    "VectorStore",
    "cache_key",
    "normalize_payload",
]
