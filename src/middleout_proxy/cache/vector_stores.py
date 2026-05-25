"""Vector stores for the L2 semantic cache.

Two implementations ship:

- ``InMemoryVectorStore`` — stdlib-only, brute-force cosine ANN. O(n) search
  per request. Fine for thousands of entries; recommended for single-machine
  deployments and tests. No persistence — entries vanish on restart.

- ``QdrantVectorStore`` — adapter for a Qdrant collection. Behind an import
  guard so ``qdrant-client`` is optional. The operator passes URL, collection
  name, and API key; the adapter delegates upsert/search/delete.

Both implement the ``VectorStore`` protocol from ``cache.l2``. The L2Cache
itself doesn't care which backend is plugged in.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]; 1 = identical direction, -1 = opposite.

    Returns 0.0 when either input is zero-vector. Callers above the L2 layer
    care about [0, 1] (semantic similarity), so the L2Cache thresholds against
    raw cosine — negatives never cross the typical 0.8+ threshold.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    a_norm = 0.0
    b_norm = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi
        a_norm += ai * ai
        b_norm += bi * bi
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return dot / (math.sqrt(a_norm) * math.sqrt(b_norm))


class InMemoryVectorStore:
    """Brute-force in-process vector store with bounded capacity + LRU eviction.

    Concurrency-safe via a single ``threading.Lock``. Search is O(n * dim) per
    call; for the L2 hot path that's acceptable up to ~10k entries (a few ms
    on modern CPUs). Operators with larger working sets should switch to
    ``QdrantVectorStore``.
    """

    def __init__(self, *, max_entries: int = 10_000) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = int(max_entries)
        self._lock = threading.Lock()
        # OrderedDict keyed by point_id; insertion order doubles as access
        # order via `move_to_end` on hit.
        self._points: OrderedDict[str, tuple[list[float], dict[str, Any]]] = OrderedDict()

    def upsert(
        self,
        point_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        with self._lock:
            if point_id in self._points:
                # Replace and bump to MRU.
                self._points.pop(point_id)
            self._points[point_id] = (list(vector), dict(payload))
            while len(self._points) > self.max_entries:
                self._points.popitem(last=False)

    def search(
        self,
        vector: list[float],
        *,
        top_k: int = 5,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        if top_k < 1:
            return []
        with self._lock:
            # Snapshot to release the lock as quickly as possible. The payload
            # dicts are shallow-copied; callers shouldn't mutate them.
            items = [(pid, vec, payload) for pid, (vec, payload) in self._points.items()]
        if not items:
            return []
        scored = [
            (pid, _cosine(vector, vec), payload) for pid, vec, payload in items
        ]
        # Highest cosine first.
        scored.sort(key=lambda triple: triple[1], reverse=True)
        results = scored[:top_k]
        # Promote the top hit to MRU so eviction prefers cold entries.
        if results:
            with self._lock:
                top_pid = results[0][0]
                if top_pid in self._points:
                    self._points.move_to_end(top_pid)
        return results

    def delete(self, point_id: str) -> None:
        with self._lock:
            self._points.pop(point_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._points)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "backend": "in_memory",
                "entries": len(self._points),
                "max_entries": self.max_entries,
            }


class QdrantVectorStore:
    """Adapter for a Qdrant collection.

    Lazy-creates the collection on first upsert if it doesn't exist
    (`collection_create_if_missing=True`). Throws clear errors when the
    optional ``qdrant-client`` package is not installed.

    Tested manually against the operator's existing Qdrant instance (per the
    user spec: `text-embedding-3-large`, 1536-dim). Tests in this repo use
    the InMemoryVectorStore — adding a network-dependent Qdrant test would
    require running a Qdrant container in CI, which is out of scope here.
    """

    # L2Cache thresholds and the margin gate both assume cosine similarity
    # in [0, 1]. Dot product / Euclidean would return raw magnitudes that
    # break those checks silently. Reject mis-configurations early.
    _SUPPORTED_DISTANCES: frozenset[str] = frozenset({"Cosine"})

    def __init__(
        self,
        *,
        url: str,
        collection: str,
        dim: int,
        api_key: str | None = None,
        timeout_s: float = 10.0,
        distance: str = "Cosine",
        collection_create_if_missing: bool = True,
    ) -> None:
        if distance not in self._SUPPORTED_DISTANCES:
            raise ValueError(
                f"QdrantVectorStore distance must be one of "
                f"{sorted(self._SUPPORTED_DISTANCES)} (the L2 threshold + margin "
                f"are calibrated for cosine), got {distance!r}."
            )
        try:
            from qdrant_client import QdrantClient  # type: ignore[import-not-found]
            from qdrant_client.http import models as qmodels  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "QdrantVectorStore requires the `qdrant-client` package. "
                "Install with: `pip install qdrant-client`."
            ) from e
        self._client = QdrantClient(url=url, api_key=api_key, timeout=timeout_s)
        self._qmodels = qmodels
        self.collection = collection
        self.dim = int(dim)

        if collection_create_if_missing:
            self._ensure_collection(distance)

    def _ensure_collection(self, distance: str) -> None:
        qm = self._qmodels
        try:
            collections = self._client.get_collections().collections
            names = {c.name for c in collections}
            if self.collection in names:
                return
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance[distance.upper()]),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Qdrant ensure_collection failed: %s: %s", type(e).__name__, e)

    def upsert(
        self,
        point_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        qm = self._qmodels
        self._client.upsert(
            collection_name=self.collection,
            points=[qm.PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    def search(
        self,
        vector: list[float],
        *,
        top_k: int = 5,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        results = self._client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
        out: list[tuple[str, float, dict[str, Any]]] = []
        for r in results:
            payload = dict(r.payload) if r.payload else {}
            out.append((str(r.id), float(r.score), payload))
        return out

    def delete(self, point_id: str) -> None:
        qm = self._qmodels
        self._client.delete(
            collection_name=self.collection,
            points_selector=qm.PointIdsList(points=[point_id]),
        )

    def stats(self) -> dict[str, Any]:
        try:
            info = self._client.get_collection(self.collection)
            return {
                "backend": "qdrant",
                "collection": self.collection,
                "entries": int(getattr(info, "points_count", 0) or 0),
                "dim": self.dim,
            }
        except Exception as e:  # noqa: BLE001
            return {
                "backend": "qdrant",
                "collection": self.collection,
                "error": f"{type(e).__name__}: {e}",
            }


__all__ = ["InMemoryVectorStore", "QdrantVectorStore"]
