"""Two-tier response cache.

Phase 2: L1 exact-match (SHA256 of normalized request payload → response).
Phase 2b/3 will add L2 semantic similarity via embeddings + ANN.

Public API
----------
- `cache_key(payload)` — canonical sha256 hex of a payload after normalization.
- `L1Cache` — SQLite-backed store with bounded entry count + LRU eviction.
- `CachedResponse` — dataclass carrying status, headers, body, and metadata.

The cache lives behind a feature flag (`BRAIN_L1_CACHE_ENABLED`) and is off by
default. Streaming requests are NOT cached in this phase — SSE chunk boundary
preservation is non-trivial and a separate follow-up.
"""

from .l1 import CachedResponse, L1Cache
from .normalize import cache_key, normalize_payload

__all__ = ["CachedResponse", "L1Cache", "cache_key", "normalize_payload"]
