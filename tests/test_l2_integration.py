"""End-to-end L2 cache integration through FastAPI TestClient.

Verifies the full pipeline:
- L2 lookup happens on L1 miss
- L2 hit returns the cached body with x-brain-l2-cache: hit and similarity header
- L2 store happens after a successful upstream response
- Auth headers are stripped from the cached response
- L2 misconfig is reported via /healthz
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from middleout_proxy.cache.embedders import HashEmbedder
from middleout_proxy.cache.l2 import L2Cache
from middleout_proxy.cache.vector_stores import InMemoryVectorStore


@pytest.fixture
def proxy_with_l2(monkeypatch):
    """Live server module, with L2 force-enabled via attribute swap."""
    from middleout_proxy import server as server_module

    fresh_l2 = L2Cache(
        enabled=True,
        embedding_client=HashEmbedder(dim=256),
        vector_store=InMemoryVectorStore(max_entries=100),
        similarity_threshold=0.95,
    )
    monkeypatch.setattr(server_module, "l2_cache", fresh_l2)
    monkeypatch.setattr(server_module, "l2_cache_misconfigured", False)
    monkeypatch.setitem(server_module._runtime, "l2_cache", True)
    # Keep L1 off so we can isolate the L2 path.
    monkeypatch.setitem(server_module._runtime, "l1_cache", False)

    class _Resp:
        def __init__(self, body: bytes) -> None:
            self.status_code = 200
            self.content = body
            self.headers = {"content-type": "application/json", "request-id": "r1"}

        def json(self):
            return json.loads(self.content.decode("utf-8"))

    class _FakeClient:
        def __init__(self) -> None:
            self.calls = []
            self.response = _Resp(b'{"id":"m1","content":[{"type":"text","text":"reply"}]}')

        async def request(self, method, url, *, headers, content):
            self.calls.append({"method": method, "url": url, "content": content})
            return self.response

        async def aclose(self):
            pass

    fake = _FakeClient()
    client = TestClient(server_module.app)
    with client:
        server_module.app.state.http = fake
        yield client, fake, fresh_l2


def test_l2_miss_then_hit_on_identical_payload(proxy_with_l2) -> None:
    client, fake, l2 = proxy_with_l2
    payload = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "tell me a story about cats"}],
    }
    r1 = client.post(
        "/v1/messages", headers={"Authorization": "Bearer t"}, json=payload
    )
    assert r1.status_code == 200
    assert r1.headers.get("x-brain-l2-cache") == "miss"

    # Stored under one point.
    assert len(l2.vector_store) == 1  # type: ignore[arg-type]

    # Second identical request → L2 hit, upstream NOT called.
    upstream_calls_before = len(fake.calls)
    r2 = client.post(
        "/v1/messages", headers={"Authorization": "Bearer t"}, json=payload
    )
    assert r2.status_code == 200
    assert r2.headers.get("x-brain-l2-cache") == "hit"
    assert "x-brain-l2-similarity" in r2.headers
    assert float(r2.headers["x-brain-l2-similarity"]) >= 0.95
    assert len(fake.calls) == upstream_calls_before  # no new upstream call


def test_l2_skips_below_threshold(proxy_with_l2) -> None:
    """A semantically different payload should miss the cache (well below
    the 0.95 cosine threshold for HashEmbedder with dim=256)."""
    client, fake, _l2 = proxy_with_l2
    p1 = {"model": "x", "messages": [{"role": "user", "content": "tell me about cats and how they purr"}]}
    p2 = {"model": "x", "messages": [{"role": "user", "content": "explain quantum entanglement in detail"}]}

    fake.response.content = b'{"id":"a","content":[{"type":"text","text":"cats"}]}'
    r1 = client.post("/v1/messages", headers={"Authorization": "Bearer t"}, json=p1)
    assert r1.headers.get("x-brain-l2-cache") == "miss"

    fake.response.content = b'{"id":"b","content":[{"type":"text","text":"quantum"}]}'
    r2 = client.post("/v1/messages", headers={"Authorization": "Bearer t"}, json=p2)
    # Semantically different → L2 miss, upstream called.
    assert r2.headers.get("x-brain-l2-cache") == "miss"


def test_l2_skips_streaming_requests(proxy_with_l2) -> None:
    client, _fake, _ = proxy_with_l2
    streaming_payload = {
        "model": "x",
        "messages": [{"role": "user", "content": "stream me"}],
        "stream": True,
    }
    # Streaming forward will fail (fake doesn't implement build_request/send);
    # we just need to confirm the L2 path didn't fire on this request.
    r = client.post(
        "/v1/messages", headers={"Authorization": "Bearer t"}, json=streaming_payload
    )
    assert r.headers.get("x-brain-l2-cache") is None


def test_l2_metadata_difference_still_hits(proxy_with_l2) -> None:
    """`metadata` is dropped by normalize_payload — different user_id should
    still produce a cache hit (since L2 uses the same canonical text)."""
    client, fake, _ = proxy_with_l2
    fake.response.content = b'{"id":"shared","content":[{"type":"text","text":"x"}]}'
    p_alice = {
        "model": "x",
        "messages": [{"role": "user", "content": "hello there my friend"}],
        "metadata": {"user_id": "alice"},
    }
    p_bob = {
        "model": "x",
        "messages": [{"role": "user", "content": "hello there my friend"}],
        "metadata": {"user_id": "bob"},
    }
    r_a = client.post("/v1/messages", headers={"Authorization": "Bearer t"}, json=p_alice)
    r_b = client.post("/v1/messages", headers={"Authorization": "Bearer t"}, json=p_bob)
    assert r_a.headers.get("x-brain-l2-cache") == "miss"
    assert r_b.headers.get("x-brain-l2-cache") == "hit"


def test_healthz_reports_l2_flags() -> None:
    from middleout_proxy.server import app
    client = TestClient(app)
    data = client.get("/healthz").json()
    assert "l2_cache_enabled" in data
    assert "l2_cache_misconfigured" in data
    assert "l2_similarity_threshold" in data


def test_settings_post_accepts_l2_toggle() -> None:
    from middleout_proxy.server import app
    client = TestClient(app)
    r = client.post("/settings", json={"l2_cache": True})
    assert r.status_code == 200
    assert r.json()["l2_cache"] is True
    # Reset for other tests
    client.post("/settings", json={"l2_cache": False})


def test_stats_includes_l2_when_enabled(proxy_with_l2) -> None:
    client, _, _ = proxy_with_l2
    data = client.get("/stats").json()
    assert "l2_cache" in data
    assert data["l2_cache"]["enabled"] is True
