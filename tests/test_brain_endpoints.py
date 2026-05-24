"""End-to-end tests for the Brain proxy endpoints wired in Phase 2/3.

Covers /preview, /metrics, /cost, /cost/reset, /providers, /cache/stats,
/cache/purge, /budget, the X-Brain-Model-Hint adapter probe, and the cost
tracker hook on /v1/messages.

Each test runs through the FastAPI TestClient with `with TestClient(...) as
client:` so the lifespan handler fires and `app.state.http` is initialized
before any request hits the proxy. Tests that need a fake upstream
monkey-patch `server_module.app.state.http` *inside* the with-block.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


# -- /preview ---------------------------------------------------------------


def test_preview_returns_size_breakdown() -> None:
    from middleout_proxy.server import app

    with TestClient(app) as client:
        payload = {
            "model": "claude-3-5-sonnet-20240620",
            "messages": [
                {"role": "user", "content": "hello " * 1000},
            ],
        }
        r = client.post("/preview", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "input_chars" in body
    assert "output_chars" in body
    assert "chars_saved" in body
    assert "pct_saved" in body
    assert "events" in body and isinstance(body["events"], list)
    assert body["input_chars"] > 0


def test_preview_rejects_invalid_json() -> None:
    from middleout_proxy.server import app

    with TestClient(app) as client:
        r = client.post(
            "/preview", content=b"not json", headers={"content-type": "application/json"}
        )
    assert r.status_code == 400
    assert "invalid JSON" in r.json()["error"]


def test_preview_rejects_non_object_body() -> None:
    from middleout_proxy.server import app

    with TestClient(app) as client:
        r = client.post("/preview", json=[1, 2, 3])
    assert r.status_code == 400
    assert "JSON object" in r.json()["error"]


# -- /metrics ---------------------------------------------------------------


def test_metrics_returns_prometheus_text() -> None:
    from middleout_proxy.server import app

    with TestClient(app) as client:
        r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers.get("content-type", "")
    body = r.text
    # Required core metric families.
    assert "middleout_requests_total" in body
    assert "# HELP middleout_requests_total" in body
    assert "# TYPE middleout_requests_total counter" in body
    assert "middleout_engine_enabled" in body


# -- /cost + /cost/reset ----------------------------------------------------


def test_cost_endpoint_starts_at_zero() -> None:
    from middleout_proxy.server import app, cost_tracker

    cost_tracker.reset()
    with TestClient(app) as client:
        r = client.get("/cost")
    assert r.status_code == 200
    body = r.json()
    assert body["total_usd"] == 0.0
    assert body["total_requests"] == 0
    assert "budget" in body


def test_cost_reset_zeros_counters() -> None:
    from middleout_proxy.server import app, cost_tracker
    from middleout_proxy.cost import RequestCost

    cost_tracker.record(RequestCost(
        provider="anthropic", model="claude-3-5-sonnet",
        input_tokens=1000, output_tokens=500, usd=0.012, matched=True,
    ))
    snap_before = cost_tracker.snapshot()
    assert snap_before["total_usd"] > 0

    with TestClient(app) as client:
        r = client.post("/cost/reset")
    assert r.status_code == 200
    assert cost_tracker.snapshot()["total_usd"] == 0.0


# -- /providers -------------------------------------------------------------


def test_providers_endpoint_lists_registered_adapters() -> None:
    from middleout_proxy.server import app

    with TestClient(app) as client:
        r = client.get("/providers")
    assert r.status_code == 200
    body = r.json()
    assert "adapters" in body
    assert "routes" in body
    # Anthropic must always be present.
    assert "anthropic" in body["adapters"]
    # OpenAI / Gemini / Ollama scaffolds register too.
    assert {"openai", "google", "ollama"} <= set(body["adapters"])


# -- X-Brain-Model-Hint -----------------------------------------------------


def test_model_hint_for_unimplemented_adapter_returns_501() -> None:
    """When the hint resolves to a not-yet-implemented adapter we return 501."""
    from middleout_proxy.server import app

    with TestClient(app) as client:
        r = client.post(
            "/v1/messages",
            headers={
                "Authorization": "Bearer t",
                "X-Brain-Model-Hint": "openai",
            },
            json={"model": "gpt-4o-mini", "messages": []},
        )
    assert r.status_code == 501
    body = r.json()
    assert body["error"]["type"] == "adapter_not_implemented"
    assert body["error"]["adapter"] == "openai"


def test_model_hint_for_anthropic_passes_through(monkeypatch) -> None:
    """An X-Brain-Model-Hint of 'anthropic' (or unset) MUST NOT trigger 501."""
    from middleout_proxy import server as server_module

    class _Resp:
        def __init__(self) -> None:
            self.status_code = 200
            self.content = b'{"id":"m1","model":"claude-3-5-sonnet","content":[]}'
            self.headers = {"content-type": "application/json"}

        def json(self):  # noqa: D401
            return json.loads(self.content.decode("utf-8"))

    class _FakeClient:
        async def request(self, *args, **kwargs):
            return _Resp()

        async def aclose(self):
            pass

    with TestClient(server_module.app) as client:
        monkeypatch.setattr(server_module.app.state, "http", _FakeClient())
        r = client.post(
            "/v1/messages",
            headers={
                "Authorization": "Bearer t",
                "X-Brain-Model-Hint": "anthropic",
            },
            json={"model": "claude-3-5-sonnet", "messages": []},
        )
    assert r.status_code == 200


# -- /cache/stats + /cache/purge -------------------------------------------


def test_cache_stats_reports_l1_l2() -> None:
    from middleout_proxy.server import app

    with TestClient(app) as client:
        r = client.get("/cache/stats")
    assert r.status_code == 200
    body = r.json()
    assert "l1" in body
    assert "l2" in body
    assert "l2_misconfigured" in body
    # L2 is a stub in this phase — always disabled.
    assert body["l2"]["enabled"] is False


def test_cache_purge_returns_count() -> None:
    from middleout_proxy.server import app

    with TestClient(app) as client:
        r = client.post("/cache/purge")
    assert r.status_code == 200
    body = r.json()
    assert "l1_cleared" in body
    # When L1 is disabled in defaults, the cleared count is 0.
    assert isinstance(body["l1_cleared"], int)
    assert body["l1_cleared"] >= 0


# -- /budget ---------------------------------------------------------------


def test_budget_returns_snapshot() -> None:
    from middleout_proxy.server import app

    with TestClient(app) as client:
        r = client.get("/budget")
    assert r.status_code == 200
    body = r.json()
    assert "chars_used" in body
    assert "tokens_used" in body
    assert "exceeded" in body


# -- /healthz advertises new fields ----------------------------------------


def test_healthz_includes_new_fields() -> None:
    from middleout_proxy.server import app

    with TestClient(app) as client:
        body = client.get("/healthz").json()
    assert "l2_cache_enabled" in body
    assert "l2_cache_misconfigured" in body
    assert "providers" in body
    assert "phase" in body
    assert isinstance(body["providers"], list)
    assert "anthropic" in body["providers"]


# -- Cost tracker integration on /v1/messages ------------------------------


def test_cost_tracking_records_known_model(monkeypatch) -> None:
    """When upstream returns a Messages payload with `usage`, the proxy records
    the cost and stamps `x-brain-cost-usd` on the response."""
    from middleout_proxy import server as server_module

    server_module.cost_tracker.reset()

    response_body = json.dumps({
        "id": "msg_1",
        "model": "claude-3-5-sonnet-20240620",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_creation_input_tokens": 200,
            "cache_read_input_tokens": 400,
        },
    }).encode("utf-8")

    class _Resp:
        status_code = 200
        content = response_body
        headers = {"content-type": "application/json"}

        def json(self):
            return json.loads(response_body.decode("utf-8"))

    class _FakeClient:
        async def request(self, *args, **kwargs):
            return _Resp()

        async def aclose(self):
            pass

    with TestClient(server_module.app) as client:
        monkeypatch.setattr(server_module.app.state, "http", _FakeClient())
        r = client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer t"},
            json={
                "model": "claude-3-5-sonnet-20240620",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert r.status_code == 200
    assert "x-brain-cost-usd" in r.headers
    cost_usd = float(r.headers["x-brain-cost-usd"])
    # 1000 in @ $3/M + 500 out @ $15/M + 200 cwrite @ $3.75/M + 400 cread @ $0.30/M
    # = 0.003 + 0.0075 + 0.00075 + 0.00012 = 0.01137
    assert cost_usd == pytest.approx(0.01137, rel=1e-3)
    snap = server_module.cost_tracker.snapshot()
    assert snap["total_requests"] == 1
    assert snap["total_usd"] == pytest.approx(0.01137, rel=1e-3)


def test_cost_tracking_unknown_model_is_unmatched(monkeypatch) -> None:
    """An unknown model id records the request but with USD=0 and matched=False."""
    from middleout_proxy import server as server_module

    server_module.cost_tracker.reset()
    response_body = json.dumps({
        "id": "msg_1",
        "model": "claude-x-experimental",
        "content": [],
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }).encode("utf-8")

    class _Resp:
        status_code = 200
        content = response_body
        headers = {"content-type": "application/json"}

        def json(self):
            return json.loads(response_body.decode("utf-8"))

    class _FakeClient:
        async def request(self, *args, **kwargs):
            return _Resp()

        async def aclose(self):
            pass

    with TestClient(server_module.app) as client:
        monkeypatch.setattr(server_module.app.state, "http", _FakeClient())
        r = client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer t"},
            json={"model": "claude-x-experimental", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    # No cost header on unmatched models.
    assert "x-brain-cost-usd" not in r.headers
    snap = server_module.cost_tracker.snapshot()
    assert snap["total_requests"] == 1
    assert snap["unmatched_requests"] == 1
    assert snap["total_usd"] == 0.0
