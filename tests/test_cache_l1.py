"""L1 exact-match cache tests.

Covers:
- key normalization drops volatile fields (metadata) but keeps everything else
- canonical encoding: dict key order doesn't change the key
- L1Cache put/get round-trip
- LRU eviction at `max_entries`
- refusal to cache 4xx/5xx responses
- refusal to cache bodies over the size cap
- auth headers never survive a put → get round-trip
- stats endpoint shape
"""

from __future__ import annotations

import json

import pytest

from middleout_proxy.cache import CachedResponse, L1Cache, cache_key, normalize_payload


# -- normalize / cache_key -----------------------------------------------------


def test_cache_key_deterministic_for_same_payload() -> None:
    p = {"model": "claude-3-5-sonnet", "messages": [{"role": "user", "content": "hi"}]}
    assert cache_key(p) == cache_key(p)


def test_cache_key_independent_of_key_order() -> None:
    p1 = {"model": "claude-3-5-sonnet", "messages": [{"role": "user", "content": "hi"}]}
    p2 = {"messages": [{"role": "user", "content": "hi"}], "model": "claude-3-5-sonnet"}
    assert cache_key(p1) == cache_key(p2)


def test_cache_key_drops_metadata() -> None:
    p1 = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    p2 = {"model": "x", "messages": [{"role": "user", "content": "hi"}], "metadata": {"user_id": "alice"}}
    p3 = {"model": "x", "messages": [{"role": "user", "content": "hi"}], "metadata": {"user_id": "bob"}}
    assert cache_key(p1) == cache_key(p2) == cache_key(p3)


def test_cache_key_differs_for_different_models() -> None:
    p1 = {"model": "claude-3-5-sonnet", "messages": []}
    p2 = {"model": "claude-3-opus", "messages": []}
    assert cache_key(p1) != cache_key(p2)


def test_cache_key_differs_for_different_messages() -> None:
    p1 = {"model": "x", "messages": [{"role": "user", "content": "hello"}]}
    p2 = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    assert cache_key(p1) != cache_key(p2)


def test_cache_key_includes_temperature() -> None:
    p1 = {"model": "x", "messages": [], "temperature": 0.1}
    p2 = {"model": "x", "messages": [], "temperature": 0.9}
    assert cache_key(p1) != cache_key(p2)


def test_cache_key_includes_stream_flag() -> None:
    p1 = {"model": "x", "messages": [], "stream": False}
    p2 = {"model": "x", "messages": [], "stream": True}
    assert cache_key(p1) != cache_key(p2)


def test_normalize_does_not_mutate_input() -> None:
    p = {"model": "x", "metadata": {"user_id": "alice"}}
    normalize_payload(p)
    assert p == {"model": "x", "metadata": {"user_id": "alice"}}


def test_non_dict_payload_hashes_distinctly() -> None:
    k1 = cache_key([1, 2, 3])  # type: ignore[arg-type]
    k2 = cache_key([4, 5, 6])  # type: ignore[arg-type]
    assert k1 != k2  # distinct non-dicts get distinct keys


# -- L1Cache ------------------------------------------------------------------


def _resp(status: int = 200, body: bytes = b'{"ok":1}', headers: dict | None = None) -> CachedResponse:
    return CachedResponse(
        status_code=status,
        headers=headers or {"content-type": "application/json"},
        body=body,
        media_type="application/json",
    )


def test_put_and_get_round_trip() -> None:
    c = L1Cache(":memory:")
    c.put("k1", _resp())
    out = c.get("k1")
    assert out is not None
    assert out.status_code == 200
    assert out.body == b'{"ok":1}'
    assert out.headers["content-type"] == "application/json"
    assert out.hit_count == 1


def test_get_miss_returns_none() -> None:
    c = L1Cache(":memory:")
    assert c.get("missing") is None


def test_hit_count_accumulates() -> None:
    c = L1Cache(":memory:")
    c.put("k1", _resp())
    c.get("k1")
    c.get("k1")
    out = c.get("k1")
    assert out is not None
    assert out.hit_count == 3


def test_lru_eviction_at_capacity() -> None:
    c = L1Cache(":memory:", max_entries=2)
    c.put("a", _resp(body=b"A"))
    c.put("b", _resp(body=b"B"))
    # Access 'a' to make 'b' the LRU
    c.get("a")
    c.put("c", _resp(body=b"C"))
    # 'b' should be evicted, 'a' and 'c' remain.
    assert c.get("a") is not None
    assert c.get("c") is not None
    assert c.get("b") is None


def test_5xx_not_cached() -> None:
    c = L1Cache(":memory:")
    c.put("k", _resp(status=500))
    assert c.get("k") is None


def test_4xx_not_cached() -> None:
    c = L1Cache(":memory:")
    c.put("k", _resp(status=401))
    c.put("k2", _resp(status=429))
    assert c.get("k") is None
    assert c.get("k2") is None


def test_oversized_body_not_cached() -> None:
    c = L1Cache(":memory:", max_body_bytes=100)
    c.put("k", _resp(body=b"x" * 200))
    assert c.get("k") is None


def test_exactly_at_size_limit_is_cached() -> None:
    c = L1Cache(":memory:", max_body_bytes=10)
    c.put("k", _resp(body=b"a" * 10))
    assert c.get("k") is not None


@pytest.mark.parametrize(
    "leaky_header",
    ["authorization", "Authorization", "x-api-key", "X-Api-Key", "anthropic-api-key", "set-cookie"],
)
def test_auth_headers_stripped_on_put(leaky_header: str) -> None:
    c = L1Cache(":memory:")
    response = _resp(headers={"content-type": "application/json", leaky_header: "secret"})
    c.put("k", response)
    out = c.get("k")
    assert out is not None
    assert leaky_header.lower() not in {k.lower() for k in out.headers}
    assert "secret" not in json.dumps(out.headers)


def test_stats_reports_entries_and_bytes() -> None:
    c = L1Cache(":memory:")
    c.put("k1", _resp(body=b"aaaa"))
    c.put("k2", _resp(body=b"bbbbbb"))
    s = c.stats()
    assert s["entries"] == 2
    assert s["body_bytes"] == 10
    assert s["max_entries"] == 10_000


def test_clear_empties_cache() -> None:
    c = L1Cache(":memory:")
    c.put("k", _resp())
    c.clear()
    assert c.get("k") is None
    assert c.stats()["entries"] == 0


def test_file_backed_persists_across_instances(tmp_path) -> None:
    db = tmp_path / "cache.db"
    c1 = L1Cache(db)
    c1.put("k", _resp(body=b"persistent"))
    c1.close()
    c2 = L1Cache(db)
    out = c2.get("k")
    assert out is not None
    assert out.body == b"persistent"
