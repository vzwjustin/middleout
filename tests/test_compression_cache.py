"""Tests for _CompressionResultCache (bounded LRU)."""
from __future__ import annotations

import threading

from middleout_proxy.compression import _CompressionResultCache


def test_lru_evicts_oldest_on_overflow():
    cache = _CompressionResultCache(max_entries=3)
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    cache.put("k3", "v3")
    cache.put("k4", "v4")  # overflow; k1 should be evicted as it was the LRU
    assert cache.get("k1") is None
    assert cache.get("k2") == "v2"
    assert cache.get("k3") == "v3"
    assert cache.get("k4") == "v4"


def test_get_returns_none_on_miss_and_increments_misses():
    cache = _CompressionResultCache(max_entries=4)
    assert cache.get("absent") is None
    assert cache.stats()["misses"] == 1
    assert cache.stats()["hits"] == 0


def test_get_returns_value_on_hit_and_refreshes_lru():
    cache = _CompressionResultCache(max_entries=2)
    cache.put("a", "A")
    cache.put("b", "B")
    # Touch "a" so it becomes most-recently-used.
    assert cache.get("a") == "A"
    # Insert "c" -> should evict "b" (LRU), not "a".
    cache.put("c", "C")
    assert cache.get("a") == "A"
    assert cache.get("b") is None
    assert cache.get("c") == "C"
    # And hit counter must reflect successful gets.
    assert cache.stats()["hits"] >= 2


def test_put_updates_existing_key_without_changing_size():
    cache = _CompressionResultCache(max_entries=4)
    cache.put("k", "v1")
    cache.put("k", "v2")
    assert cache.stats()["size"] == 1
    assert cache.get("k") == "v2"


def test_disabled_cache_max_entries_zero_always_returns_none():
    cache = _CompressionResultCache(max_entries=0)
    cache.put("k", "v")
    assert cache.get("k") is None
    stats = cache.stats()
    assert stats["size"] == 0
    assert stats["max_entries"] == 0
    # Disabled get/put short-circuit before hit/miss counters fire.
    assert stats["hits"] == 0
    assert stats["misses"] == 0


def test_cache_stats_shape():
    cache = _CompressionResultCache(max_entries=8)
    cache.put("k", "v")
    cache.get("k")
    cache.get("missing")
    stats = cache.stats()
    for key in ("size", "max_entries", "hits", "misses"):
        assert key in stats
    assert stats["size"] == 1
    assert stats["max_entries"] == 8
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_cache_thread_safe_under_concurrent_put():
    cache = _CompressionResultCache(max_entries=50)
    errors: list[BaseException] = []

    def worker(start: int):
        try:
            for i in range(100):
                cache.put(f"thread{start}-{i}", f"value-{i}")
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # Final size must not exceed the configured cap.
    assert cache.stats()["size"] <= 50
