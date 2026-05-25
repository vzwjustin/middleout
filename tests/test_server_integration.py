"""FastAPI integration tests for middleout-proxy's server."""
from __future__ import annotations

import copy
from pathlib import Path
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

# Import the server module lazily so the conftest autouse fixture has a chance
# to scrub BLOCKED_AUTH_ENV_VARS before any module-level `load_settings()` runs.
# (Module-level code in server.py is only re-executed once across the suite.)
from middleout_proxy import server as server_module


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    """Boot a FastAPI TestClient against the proxy app and isolate runtime state.

    Snapshots `server._runtime` and redirects `_RUNTIME_PERSIST_PATH` to `tmp_path`
    so each test starts/ends with the same in-memory state and never writes to
    the real `.middleout-logs/runtime_settings.json`.
    """
    snapshot = copy.deepcopy(server_module._runtime)
    monkeypatch.setattr(
        server_module, "_RUNTIME_PERSIST_PATH", tmp_path / "runtime_settings.json"
    )
    with TestClient(server_module.app) as c:
        yield c
    # Restore in-memory runtime state.
    server_module._runtime.clear()
    server_module._runtime.update(snapshot)


# ---------------------------------------------------------------------------
# Local-only endpoints (no upstream calls)
# ---------------------------------------------------------------------------

def test_healthz_returns_ok_and_subscription_only_contract(client: TestClient):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["api_keys_supported"] is False
    assert body["api_key_injection"] is False
    assert body["api_key_headers_rejected"] is True
    assert body["auth_mode"] == "subscription_oauth_passthrough_only"


def test_stats_returns_basic_counter_shape(client: TestClient):
    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "requests_total" in body
    assert "uptime_s" in body
    assert "result_cache" in body
    # result_cache shape from compressor.result_cache.stats().
    assert {"size", "max_entries", "hits", "misses"} <= set(body["result_cache"].keys())


def test_settings_get_returns_runtime_dict(client: TestClient):
    resp = client.get("/settings")
    assert resp.status_code == 200
    body = resp.json()
    # The runtime config dict carries these top-level keys.
    for key in ("input_compression", "output_compression", "jl_dedupe", "caveman", "rtk"):
        assert key in body
    assert isinstance(body["caveman"], dict)
    assert isinstance(body["rtk"], dict)


def test_settings_post_flips_input_compression_and_persists(client: TestClient):
    resp = client.post("/settings", json={"input_compression": False})
    assert resp.status_code == 200
    assert resp.json()["input_compression"] is False
    # Persisted in memory and via subsequent GET.
    follow_up = client.get("/settings").json()
    assert follow_up["input_compression"] is False


def test_settings_post_updates_nested_caveman(client: TestClient):
    resp = client.post(
        "/settings", json={"caveman": {"enabled": True, "level": "aggressive"}}
    )
    assert resp.status_code == 200
    cv = resp.json()["caveman"]
    assert cv["enabled"] is True
    assert cv["level"] == "aggressive"
    # Persisted.
    follow_up = client.get("/settings").json()
    assert follow_up["caveman"]["level"] == "aggressive"


def test_settings_post_invalid_level_returns_400(client: TestClient):
    resp = client.post("/settings", json={"caveman": {"level": "garbage"}})
    assert resp.status_code == 400
    body = resp.json()
    assert "error" in body
    assert "caveman level must be one of" in body["error"]


def test_settings_post_caveman_not_a_dict_returns_400(client: TestClient):
    resp = client.post("/settings", json={"caveman": "not-a-dict"})
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] == "caveman must be an object"


def test_dashboard_returns_html(client: TestClient):
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "").lower()
    assert "MiddleOut" in resp.text


# ---------------------------------------------------------------------------
# Strict auth gate on the catch-all proxy path
# ---------------------------------------------------------------------------

def test_proxy_path_without_authorization_returns_401(client: TestClient):
    resp = client.post(
        "/v1/messages",
        json={"model": "claude-3-5-sonnet", "max_tokens": 100, "messages": []},
    )
    assert resp.status_code == 401
    body = resp.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "strict_subscription_auth_error"


def test_proxy_path_with_x_api_key_returns_401(client: TestClient):
    resp = client.post(
        "/v1/messages",
        json={"model": "claude-3-5-sonnet", "max_tokens": 100, "messages": []},
        headers={
            "authorization": "Bearer oauth-token-from-claude-code-login",
            "x-api-key": "leaked-api-key-should-be-rejected",
        },
    )
    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["type"] == "strict_subscription_auth_error"


# ---------------------------------------------------------------------------
# Catch-all root metadata endpoint
# ---------------------------------------------------------------------------

def test_catch_all_root_returns_metadata_json(client: TestClient):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "middleout-claude-proxy"
    assert body["health"] == "/healthz"
    assert body["stats"] == "/stats"
    assert body["anthropic_messages"] == "/v1/messages"
