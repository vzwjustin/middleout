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


def test_cost_reset_also_resets_budget() -> None:
    """`/cost/reset` should clear the budget counters in the same shot.

    Operators rolling over a billing period don't want stale ``chars_used``
    values lingering after a reset — that was the bug a recent live test
    exposed.
    """
    from middleout_proxy.server import app, cost_tracker, usage_budget
    from middleout_proxy.cost import RequestCost

    cost_tracker.record(RequestCost(
        provider="anthropic", model="claude-3-5-sonnet",
        input_tokens=1, output_tokens=1, usd=0.01, matched=True,
    ))
    usage_budget.record(chars=12345, tokens=678)
    assert usage_budget.snapshot()["chars_used"] == 12345
    assert usage_budget.snapshot()["tokens_used"] == 678

    with TestClient(app) as client:
        r = client.post("/cost/reset")
    assert r.status_code == 200
    body = r.json()
    assert body == {"reset": True, "total_usd": 0.0, "budget_reset": True}

    assert cost_tracker.snapshot()["total_usd"] == 0.0
    assert usage_budget.snapshot()["chars_used"] == 0
    assert usage_budget.snapshot()["tokens_used"] == 0


def test_budget_reset_endpoint_zeroes_budget_only() -> None:
    """`/budget/reset` clears budget counters but keeps cost-tracker rows."""
    from middleout_proxy.server import app, cost_tracker, usage_budget
    from middleout_proxy.cost import RequestCost

    cost_tracker.record(RequestCost(
        provider="anthropic", model="claude-3-5-sonnet",
        input_tokens=1, output_tokens=1, usd=0.05, matched=True,
    ))
    usage_budget.record(chars=999, tokens=99)

    with TestClient(app) as client:
        r = client.post("/budget/reset")
    assert r.status_code == 200
    body = r.json()
    assert body["reset"] is True
    assert body["budget"]["chars_used"] == 0
    assert body["budget"]["tokens_used"] == 0
    # cost tracker must be untouched
    assert cost_tracker.snapshot()["total_usd"] > 0


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


def test_audit_log_captures_request_model(monkeypatch) -> None:
    """Each upstream-successful request must tag its audit row with the request body's model."""
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
            headers={"Authorization": "Bearer t"},
            json={"model": "claude-3-5-sonnet-latest", "messages": []},
        )
        assert r.status_code == 200
        recent = client.get("/stats/recent?n=1").json()
    items = recent.get("items") or []
    assert len(items) >= 1
    assert items[-1]["model"] == "claude-3-5-sonnet-latest"


def test_brain_engine_headers_emitted_on_compressed_response(monkeypatch) -> None:
    """When compression actually fires, the response carries `x-brain-engines`
    (per-engine breakdown) and `x-brain-chars-saved-in` (alias for the legacy
    `x-middleout-input-chars-saved` header).
    """
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

    # A request large + repetitive enough for at least one engine to fire.
    msg = "the quick brown fox jumps over the lazy dog please just basically " * 200
    with TestClient(server_module.app) as client:
        monkeypatch.setattr(server_module.app.state, "http", _FakeClient())
        # Make sure the legacy engines are on (default is on, but be explicit).
        client.post("/settings", json={"caveman": {"enabled": True, "level": "standard"}})
        r = client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer t"},
            json={
                "model": "claude-3-5-sonnet",
                "messages": [{"role": "user", "content": msg}],
            },
        )
    assert r.status_code == 200
    # Brain-prefixed alias header should always come along with the legacy one.
    assert "x-brain-chars-saved-in" in r.headers
    assert r.headers["x-brain-chars-saved-in"] == r.headers["x-middleout-input-chars-saved"]
    # Per-engine line is only set when at least one engine saved bytes — for
    # this oversized verbose-English payload, caveman should definitely fire.
    if int(r.headers["x-brain-chars-saved-in"]) > 0:
        assert "x-brain-engines" in r.headers
        # Format: comma-separated `name=savedBytes`
        for part in r.headers["x-brain-engines"].split(","):
            k, _, v = part.partition("=")
            assert k.strip(), part
            assert v.strip().isdigit(), part


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


# -- runtime persistence sync ----------------------------------------------


def test_persisted_l2_runtime_flag_syncs_l2cache_enabled(tmp_path) -> None:
    """Persisted `l2_cache: true` must sync onto the live L2Cache.enabled.

    Regression: a process that boots with BRAIN_L2_CACHE_ENABLED unset (so
    `l2_cache.enabled == False` at construction) but a persisted runtime
    snapshot of `l2_cache: true` would report `True` from /settings but
    `False` from /healthz, and L2 lookups would be silently skipped.

    Uses a subprocess so module-level side effects don't leak into other
    tests in this session.
    """
    import json
    import subprocess
    import sys

    (tmp_path / "runtime_settings.json").write_text(
        json.dumps({
            "input_compression": True,
            "output_compression": True,
            "l2_cache": True,
        })
    )

    script = (
        "import json\n"
        "import middleout_proxy.server as srv\n"
        "from fastapi.testclient import TestClient\n"
        "out = {\n"
        "    'runtime_l2': srv._runtime.get('l2_cache'),\n"
        "    'l2cache_obj_exists': srv.l2_cache is not None,\n"
        "    'l2cache_obj_enabled': srv.l2_cache.enabled if srv.l2_cache else None,\n"
        "}\n"
        "with TestClient(srv.app) as client:\n"
        "    out['healthz_l2'] = client.get('/healthz').json()['l2_cache_enabled']\n"
        "    out['settings_l2'] = client.get('/settings').json()['l2_cache']\n"
        "print(json.dumps(out))\n"
    )

    env = {
        "PATH": __import__("os").environ.get("PATH", ""),
        "PYTHONPATH": __import__("os").environ.get("PYTHONPATH", "src"),
        "MIDDLEOUT_AUDIT_DIR": str(tmp_path),
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    out = json.loads(result.stdout.strip().splitlines()[-1])

    assert out["runtime_l2"] is True
    assert out["l2cache_obj_exists"] is True
    assert out["l2cache_obj_enabled"] is True, (
        "L2Cache.enabled must follow the persisted runtime flag at boot, "
        "not the (likely-False) env-driven default"
    )
    assert out["healthz_l2"] is True
    assert out["settings_l2"] is True


# -- /rate-limit + policy router wiring (Phase 4) --------------------------


def test_rate_limit_endpoint_returns_state() -> None:
    """`/rate-limit` exposes the token-bucket bookkeeping plus the runtime flag."""
    from middleout_proxy.server import app

    with TestClient(app) as client:
        r = client.get("/rate-limit")
    assert r.status_code == 200
    body = r.json()
    assert "enabled" in body
    assert "active_buckets" in body
    assert "capacity" in body
    assert "refill_per_second" in body
    assert "policies_rules" in body
    assert "policies_default_is_vanilla" in body
    # Defaults: capacity is positive, refill is positive.
    assert body["capacity"] > 0
    assert body["refill_per_second"] > 0


def test_rate_limit_returns_429_when_exhausted(monkeypatch) -> None:
    """When rate_limit is enabled, exhausting the bucket yields a 429 with
    `x-brain-rate-limit: exceeded` and a `retry-after` header. Bucket key is a
    hash of the Authorization header, so raw tokens never reach the limiter.
    """
    from middleout_proxy import server as server_module

    class _Resp:
        status_code = 200
        content = b'{"id":"m1","model":"claude-3-5-sonnet","content":[]}'
        headers = {"content-type": "application/json"}

        def json(self):  # noqa: D401
            import json as _j

            return _j.loads(self.content.decode("utf-8"))

    class _FakeClient:
        async def request(self, *args, **kwargs):
            return _Resp()

        async def aclose(self):
            pass

    # Swap in a tiny bucket so we don't need 60+ requests to exhaust it.
    from middleout_proxy.rate_limit import RequestLimiter

    original_limiter = server_module.request_limiter
    tiny = RequestLimiter(capacity=2, refill_per_second=0.001)
    monkeypatch.setattr(server_module, "request_limiter", tiny)
    monkeypatch.setitem(server_module._runtime, "rate_limit", True)
    try:
        with TestClient(server_module.app) as client:
            monkeypatch.setattr(server_module.app.state, "http", _FakeClient())
            # First two go through.
            for _ in range(2):
                r = client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer t"},
                    json={"model": "claude-3-5-sonnet", "messages": []},
                )
                assert r.status_code == 200
            # Third one exhausts the bucket.
            r = client.post(
                "/v1/messages",
                headers={"Authorization": "Bearer t"},
                json={"model": "claude-3-5-sonnet", "messages": []},
            )
            assert r.status_code == 429
            assert r.headers["x-brain-rate-limit"] == "exceeded"
            assert "retry-after" in r.headers
            body = r.json()
            assert body["error"]["type"] == "rate_limit_error"
    finally:
        monkeypatch.setattr(server_module, "request_limiter", original_limiter)


def test_rate_limit_off_by_default_passes_through(monkeypatch) -> None:
    """When rate_limit toggle is off, an empty bucket is irrelevant — all
    traffic flows through to upstream.
    """
    from middleout_proxy import server as server_module

    class _Resp:
        status_code = 200
        content = b'{"id":"m1","model":"claude-3-5-sonnet","content":[]}'
        headers = {"content-type": "application/json"}

        def json(self):  # noqa: D401
            import json as _j

            return _j.loads(self.content.decode("utf-8"))

    class _FakeClient:
        async def request(self, *args, **kwargs):
            return _Resp()

        async def aclose(self):
            pass

    from middleout_proxy.rate_limit import RequestLimiter

    original_limiter = server_module.request_limiter
    tiny = RequestLimiter(capacity=1, refill_per_second=0.001)
    monkeypatch.setattr(server_module, "request_limiter", tiny)
    monkeypatch.setitem(server_module._runtime, "rate_limit", False)
    try:
        with TestClient(server_module.app) as client:
            monkeypatch.setattr(server_module.app.state, "http", _FakeClient())
            # 5 requests, well over capacity — all should pass.
            for _ in range(5):
                r = client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer t"},
                    json={"model": "claude-3-5-sonnet", "messages": []},
                )
                assert r.status_code == 200
    finally:
        monkeypatch.setattr(server_module, "request_limiter", original_limiter)


def test_policies_endpoint_returns_rules_and_default() -> None:
    """`/policies` exposes the PolicyRouter's loaded rules and default policy."""
    from middleout_proxy.server import app

    with TestClient(app) as client:
        r = client.get("/policies")
    assert r.status_code == 200
    body = r.json()
    assert "rules" in body and isinstance(body["rules"], list)
    assert "default" in body and isinstance(body["default"], dict)
    # Default policy advertises the expected fields.
    for k in (
        "input_compression",
        "output_compression",
        "jl_dedupe",
        "caveman_enabled",
        "caveman_level",
        "rtk_enabled",
        "rtk_level",
        "max_text_chars",
    ):
        assert k in body["default"]


def test_policy_router_overrides_runtime_settings(monkeypatch) -> None:
    """A model-glob policy match overrides the runtime engine settings for
    that request only, without persisting any change.
    """
    from middleout_proxy import server as server_module
    from middleout_proxy.policies import (
        CompressionPolicy,
        PolicyMatch,
        PolicyRouter,
    )

    captured: dict[str, bytes | None] = {"content": None}

    class _Resp:
        status_code = 200
        content = b'{"id":"m1","model":"claude-3-5-sonnet","content":[]}'
        headers = {"content-type": "application/json"}

        def json(self):  # noqa: D401
            import json as _j

            return _j.loads(self.content.decode("utf-8"))

    class _FakeClient:
        async def request(self, *args, **kwargs):
            captured["content"] = kwargs.get("content")
            return _Resp()

        async def aclose(self):
            pass

    # Policy: disable all compression for `claude-3-5-haiku*`. The body bytes
    # the proxy forwards upstream must therefore equal the original request
    # bytes — no compression applied.
    no_compression = CompressionPolicy(
        input_compression=False,
        output_compression=False,
        jl_dedupe=False,
        caveman_enabled=False,
        rtk_enabled=False,
    )
    router = PolicyRouter(
        rules=[
            PolicyMatch(
                model_glob="claude-3-5-haiku*",
                endpoint="v1/messages",
                policy=no_compression,
            ),
        ],
    )
    monkeypatch.setattr(server_module, "policy_router", router)

    # Force the runtime to have engines fully ON, so the only way the request
    # body passes through unchanged is if the policy override fires.
    monkeypatch.setitem(server_module._runtime, "input_compression", True)

    big_payload = {
        "model": "claude-3-5-haiku-latest",
        "messages": [
            {
                "role": "user",
                "content": "the quick brown fox " * 500,
            }
        ],
    }
    import json as _json

    expected_bytes = _json.dumps(big_payload).encode("utf-8")

    with TestClient(server_module.app) as client:
        monkeypatch.setattr(server_module.app.state, "http", _FakeClient())
        r = client.post(
            "/v1/messages",
            headers={
                "Authorization": "Bearer t",
                "Content-Type": "application/json",
            },
            content=expected_bytes,
        )
    assert r.status_code == 200
    # Policy turned off input_compression for this model — proxy must forward
    # the original bytes verbatim.
    assert captured["content"] == expected_bytes


def test_policies_default_unchanged_when_no_rules() -> None:
    """With an empty rules list and a vanilla default policy, behavior matches
    the legacy `_runtime` toggle path. /policies just reports the defaults.
    """
    from middleout_proxy.server import app
    from middleout_proxy import server as server_module
    from middleout_proxy.policies import CompressionPolicy, PolicyRouter

    # Restore a vanilla router for this test (auto-fixture restores after).
    vanilla = PolicyRouter(rules=[], default=CompressionPolicy())
    server_module.policy_router = vanilla

    with TestClient(app) as client:
        body = client.get("/policies").json()
    assert body["rules"] == []
    # The vanilla default has all engines on, max_text_chars sized for sanity.
    assert body["default"]["input_compression"] is True
    assert body["default"]["output_compression"] is False
    assert body["default"]["jl_dedupe"] is True


def test_healthz_advertises_rate_limit_state() -> None:
    """`/healthz` reports `rate_limit_*` so operators can confirm wiring."""
    from middleout_proxy.server import app

    with TestClient(app) as client:
        body = client.get("/healthz").json()
    assert "rate_limit_enabled" in body
    assert "rate_limit_capacity" in body
    assert "rate_limit_refill_per_second" in body
    assert isinstance(body["rate_limit_enabled"], bool)
    assert body["rate_limit_capacity"] > 0
    assert body["rate_limit_refill_per_second"] > 0


