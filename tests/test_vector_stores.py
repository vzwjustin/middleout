"""InMemoryVectorStore tests."""

from __future__ import annotations

import math

import pytest

from middleout_proxy.cache.vector_stores import InMemoryVectorStore, _cosine


# -- _cosine -----------------------------------------------------------------


def test_cosine_identical_vectors_is_one() -> None:
    v = [1.0, 2.0, 3.0]
    assert math.isclose(_cosine(v, v), 1.0)


def test_cosine_orthogonal_is_zero() -> None:
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert math.isclose(_cosine(a, b), 0.0)


def test_cosine_opposite_is_negative_one() -> None:
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert math.isclose(_cosine(a, b), -1.0)


def test_cosine_mismatched_dims_returns_zero() -> None:
    assert _cosine([1.0, 0.0], [1.0]) == 0.0


def test_cosine_empty_vectors_returns_zero() -> None:
    assert _cosine([], []) == 0.0


def test_cosine_zero_vector_returns_zero() -> None:
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


# -- InMemoryVectorStore -----------------------------------------------------


def test_upsert_and_search() -> None:
    s = InMemoryVectorStore(max_entries=10)
    s.upsert("p1", [1.0, 0.0, 0.0], {"data": "a"})
    s.upsert("p2", [0.0, 1.0, 0.0], {"data": "b"})
    results = s.search([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0][0] == "p1"
    assert math.isclose(results[0][1], 1.0)
    assert results[1][0] == "p2"


def test_search_empty_store_returns_empty() -> None:
    s = InMemoryVectorStore()
    assert s.search([1.0, 0.0]) == []


def test_search_top_k_caps_results() -> None:
    s = InMemoryVectorStore()
    for i in range(5):
        s.upsert(f"p{i}", [float(i), 0.0], {"i": i})
    assert len(s.search([1.0, 0.0], top_k=3)) == 3
    assert len(s.search([1.0, 0.0], top_k=0)) == 0


def test_upsert_replaces_same_id() -> None:
    s = InMemoryVectorStore()
    s.upsert("p1", [1.0, 0.0], {"v": "old"})
    s.upsert("p1", [0.0, 1.0], {"v": "new"})
    assert len(s) == 1
    results = s.search([0.0, 1.0], top_k=1)
    assert results[0][2]["v"] == "new"


def test_delete_removes_entry() -> None:
    s = InMemoryVectorStore()
    s.upsert("p1", [1.0, 0.0], {})
    assert len(s) == 1
    s.delete("p1")
    assert len(s) == 0
    s.delete("does-not-exist")  # No exception.


def test_max_entries_lru_eviction() -> None:
    s = InMemoryVectorStore(max_entries=3)
    s.upsert("a", [1.0, 0.0], {})
    s.upsert("b", [0.0, 1.0], {})
    s.upsert("c", [1.0, 1.0], {})
    # Access "a" → moves to MRU on search.
    s.search([1.0, 0.0], top_k=1)
    # Insert "d" → "b" (now LRU) should be evicted.
    s.upsert("d", [0.5, 0.5], {})
    assert len(s) == 3
    ids = {pid for pid, _, _ in s.search([1.0, 1.0], top_k=5)}
    assert "a" in ids
    assert "c" in ids
    assert "d" in ids
    assert "b" not in ids


def test_stats_reports_backend() -> None:
    s = InMemoryVectorStore(max_entries=42)
    s.upsert("p", [1.0], {})
    stats = s.stats()
    assert stats["backend"] == "in_memory"
    assert stats["entries"] == 1
    assert stats["max_entries"] == 42


def test_max_entries_validates() -> None:
    with pytest.raises(ValueError):
        InMemoryVectorStore(max_entries=0)
