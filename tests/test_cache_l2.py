"""Unit tests for the L2 semantic cache stub (Phase 2b)."""

from __future__ import annotations

from typing import Any

import pytest

from middleout_proxy.cache.l1 import CachedResponse
from middleout_proxy.cache.l2 import (
    L2Cache,
    L2NotConfigured,
)


class _StubEmbedder:
    dim = 4

    def __init__(self, vec: list[float] | None = None) -> None:
        self.vec = vec or [0.1, 0.2, 0.3, 0.4]
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        return list(self.vec)


class _StubVectorStore:
    def __init__(self) -> None:
        self.upserts: list[tuple[str, list[float], dict[str, Any]]] = []
        self.results: list[tuple[str, float, dict[str, Any]]] = []

    def upsert(self, point_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        self.upserts.append((point_id, vector, payload))

    def search(self, vector: list[float], *, top_k: int = 5) -> list[tuple[str, float, dict[str, Any]]]:
        return self.results[:top_k]

    def delete(self, point_id: str) -> None:
        pass


# -- construction ---------------------------------------------------------


def test_l2_disabled_by_default() -> None:
    cache = L2Cache()
    assert cache.enabled is False
    assert cache.lookups == 0
    assert cache.hits == 0


def test_l2_enabled_without_client_raises_l2_not_configured() -> None:
    with pytest.raises(L2NotConfigured):
        L2Cache(enabled=True)
    with pytest.raises(L2NotConfigured):
        L2Cache(enabled=True, embedding_client=_StubEmbedder())  # no vector store
    with pytest.raises(L2NotConfigured):
        L2Cache(enabled=True, vector_store=_StubVectorStore())  # no embedder


def test_l2_enabled_with_full_wiring_works() -> None:
    cache = L2Cache(
        enabled=True,
        embedding_client=_StubEmbedder(),
        vector_store=_StubVectorStore(),
    )
    assert cache.enabled is True


# -- get_similar ---------------------------------------------------------


def test_get_similar_returns_none_when_disabled() -> None:
    cache = L2Cache(enabled=False)
    assert cache.get_similar("anything") is None
    assert cache.lookups == 0


def test_get_similar_returns_hit_when_above_threshold() -> None:
    embedder = _StubEmbedder()
    store = _StubVectorStore()
    # Plant a result that exactly equals the threshold so the hit is reported.
    # The L2 stub deserializes from raw metadata bytes, so we construct the
    # metadata payload directly rather than round-tripping a CachedResponse.
    _ = CachedResponse  # imported for type-availability assertion
    import base64
    store.results = [
        (
            "pt_a",
            0.99,
            {
                "status_code": 200,
                "headers": {"content-type": "application/json"},
                "body_b64": base64.b64encode(b'{"id":"m1"}').decode("ascii"),
                "media_type": "application/json",
                "inserted_at": 0.0,
                "hit_count": 0,
            },
        )
    ]
    cache = L2Cache(
        enabled=True,
        embedding_client=embedder,
        vector_store=store,
        similarity_threshold=0.97,
    )
    hit = cache.get_similar("hello world")
    assert hit is not None
    assert hit.similarity == 0.99
    assert hit.point_id == "pt_a"
    assert hit.response.status_code == 200
    assert hit.response.body == b'{"id":"m1"}'
    assert cache.lookups == 1
    assert cache.hits == 1


def test_get_similar_misses_when_below_threshold() -> None:
    store = _StubVectorStore()
    store.results = [("pt_a", 0.50, {"status_code": 200, "headers": {}, "body_b64": ""})]
    cache = L2Cache(
        enabled=True,
        embedding_client=_StubEmbedder(),
        vector_store=store,
        similarity_threshold=0.90,
    )
    assert cache.get_similar("hello") is None
    assert cache.lookups == 1
    assert cache.hits == 0


def test_get_similar_swallows_embedding_errors() -> None:
    class _Broken(_StubEmbedder):
        def embed(self, text: str) -> list[float]:
            raise RuntimeError("embed failed")

    cache = L2Cache(
        enabled=True,
        embedding_client=_Broken(),
        vector_store=_StubVectorStore(),
    )
    assert cache.get_similar("anything") is None
    # The lookup attempt was counted before the embedder raised.
    assert cache.lookups == 1
    assert cache.hits == 0


def test_get_similar_respects_caller_threshold_override() -> None:
    store = _StubVectorStore()
    store.results = [("pt_a", 0.85, {
        "status_code": 200, "headers": {}, "body_b64": "",
    })]
    cache = L2Cache(
        enabled=True,
        embedding_client=_StubEmbedder(),
        vector_store=store,
        similarity_threshold=0.97,  # default would miss
    )
    # Caller-supplied threshold lower than the default — hit.
    hit = cache.get_similar("anything", threshold=0.80)
    assert hit is not None
    assert hit.similarity == 0.85


# -- put_similar ---------------------------------------------------------


def test_put_similar_no_op_when_disabled() -> None:
    store = _StubVectorStore()
    cache = L2Cache(enabled=False, embedding_client=_StubEmbedder(), vector_store=store)
    response = CachedResponse(status_code=200, headers={}, body=b"")
    cache.put_similar("text", response, point_id="pt_a")
    assert store.upserts == []


def test_put_similar_writes_when_enabled() -> None:
    embedder = _StubEmbedder()
    store = _StubVectorStore()
    cache = L2Cache(enabled=True, embedding_client=embedder, vector_store=store)
    response = CachedResponse(
        status_code=200,
        headers={"content-type": "application/json"},
        body=b'{"id":"m1"}',
        media_type="application/json",
    )
    cache.put_similar("hello", response, point_id="pt_a")
    assert len(store.upserts) == 1
    point_id, vector, payload = store.upserts[0]
    assert point_id == "pt_a"
    assert vector == embedder.vec
    assert payload["status_code"] == 200
    # Body is round-trippable.
    import base64
    assert base64.b64decode(payload["body_b64"]) == b'{"id":"m1"}'


def test_put_similar_swallows_errors() -> None:
    class _BrokenStore(_StubVectorStore):
        def upsert(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("upsert failed")

    cache = L2Cache(
        enabled=True, embedding_client=_StubEmbedder(), vector_store=_BrokenStore()
    )
    response = CachedResponse(status_code=200, headers={}, body=b"")
    # Must not raise.
    cache.put_similar("anything", response, point_id="pt_a")


# -- stats ----------------------------------------------------------------


def test_l2_stats_shape() -> None:
    cache = L2Cache(enabled=True, embedding_client=_StubEmbedder(), vector_store=_StubVectorStore())
    stats = cache.stats()
    assert stats["enabled"] is True
    assert stats["lookups"] == 0
    assert stats["hits"] == 0
    assert stats["embedding_dim"] == 4
