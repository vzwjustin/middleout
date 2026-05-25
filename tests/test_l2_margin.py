"""L2 semantic cache margin gate.

When several candidates are above threshold, the top match must beat the
runner-up by `margin`. This defends against near-ties (a long shared prefix
pushing unrelated prompts above 0.97) silently returning the wrong cached
response.
"""
from __future__ import annotations

from middleout_proxy.cache.l2 import L2Cache


class _FixedEmbedder:
    """Embedding stub — every call returns the same vector."""

    dim = 4

    def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return [1.0, 0.0, 0.0, 0.0]


class _StubStore:
    """Returns canned (id, similarity, payload) triples regardless of vector."""

    def __init__(self, results: list[tuple[str, float, dict]]) -> None:
        self._results = results

    def upsert(self, point_id, vector, payload):  # noqa: ARG002
        pass

    def search(self, vector, *, top_k):  # noqa: ARG002
        return self._results[:top_k]

    def delete(self, point_id):  # noqa: ARG002
        pass


def _cached() -> dict:
    import base64

    return {
        "status_code": 200,
        "headers": {"content-type": "application/json"},
        "body_b64": base64.b64encode(b'{"id":"m1"}').decode("ascii"),
        "media_type": "application/json",
        "inserted_at": 0.0,
        "hit_count": 0,
    }


def _make(results, *, margin=0.02, threshold=0.97):
    return L2Cache(
        embedding_client=_FixedEmbedder(),
        vector_store=_StubStore(results),
        similarity_threshold=threshold,
        top_k=5,
        margin=margin,
        enabled=True,
    )


def test_single_match_above_threshold_hits():
    cache = _make([("a", 0.99, _cached())])
    hit = cache.get_similar("any")
    assert hit is not None
    assert hit.similarity == 0.99


def test_top_below_threshold_misses():
    cache = _make([("a", 0.50, _cached())])
    assert cache.get_similar("any") is None


def test_near_tie_above_threshold_is_a_miss():
    """0.99 vs 0.98 — margin 0.02 not met, refuse the lookup."""
    cache = _make([("a", 0.99, _cached()), ("b", 0.98, _cached())], margin=0.02)
    assert cache.get_similar("any") is None


def test_clear_winner_above_margin_hits():
    """0.99 vs 0.80 — margin 0.02 cleared, serve the top match."""
    cache = _make([("a", 0.99, _cached()), ("b", 0.80, _cached())], margin=0.02)
    hit = cache.get_similar("any")
    assert hit is not None
    assert hit.point_id == "a"


def test_margin_zero_disables_the_gate():
    cache = _make([("a", 0.99, _cached()), ("b", 0.989, _cached())], margin=0.0)
    hit = cache.get_similar("any")
    assert hit is not None


def test_runner_up_below_threshold_does_not_block_top():
    """Only the candidates above threshold matter for the margin gate."""
    cache = _make([("a", 0.99, _cached()), ("b", 0.10, _cached())], margin=0.02)
    hit = cache.get_similar("any")
    assert hit is not None


def test_stats_includes_top_k_and_margin():
    cache = _make([], margin=0.03)
    stats = cache.stats()
    assert stats["top_k"] == 5
    assert stats["margin"] == 0.03
