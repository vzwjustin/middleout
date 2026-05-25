"""Admin endpoint auth gate.

When `MIDDLEOUT_ADMIN_TOKEN` is unset, admin routes are open as before
(loopback-only deployment). When set, every admin route requires a matching
`Authorization: Bearer <token>` and returns 401 otherwise.

The catch-all proxy `/{path:path}` and `/healthz` are unaffected — those
have their own auth contract (strict subscription Bearer OAuth) handled in
`_forward_request_headers`.
"""
from __future__ import annotations

import dataclasses

import pytest
from fastapi.testclient import TestClient


def _patch_admin_token(monkeypatch, server_module, token: str) -> None:
    """Replace the module-level frozen Settings with one carrying `admin_token`."""
    new_settings = dataclasses.replace(server_module.settings, admin_token=token)
    monkeypatch.setattr(server_module, "settings", new_settings)


@pytest.fixture
def client_no_admin_token(monkeypatch):
    """Default deployment: admin endpoints open."""
    from middleout_proxy import server as server_module

    _patch_admin_token(monkeypatch, server_module, "")
    with TestClient(server_module.app) as c:
        yield c


@pytest.fixture
def client_with_admin_token(monkeypatch):
    """Operator set MIDDLEOUT_ADMIN_TOKEN; admin endpoints must be authed."""
    from middleout_proxy import server as server_module

    _patch_admin_token(monkeypatch, server_module, "s3cret")
    with TestClient(server_module.app) as c:
        yield c


def test_admin_token_unset_settings_is_open(client_no_admin_token):
    r = client_no_admin_token.get("/settings")
    assert r.status_code == 200


def test_admin_token_set_blocks_without_authz(client_with_admin_token):
    r = client_with_admin_token.get("/settings")
    assert r.status_code == 401
    body = r.json()
    assert body["error"]["type"] == "admin_auth_error"


def test_admin_token_set_blocks_with_wrong_token(client_with_admin_token):
    r = client_with_admin_token.get(
        "/settings", headers={"Authorization": "Bearer wrong"}
    )
    assert r.status_code == 401


def test_admin_token_set_blocks_with_non_bearer_authz(client_with_admin_token):
    r = client_with_admin_token.get(
        "/settings", headers={"Authorization": "Basic s3cret"}
    )
    assert r.status_code == 401


def test_admin_token_set_allows_correct_token(client_with_admin_token):
    r = client_with_admin_token.get(
        "/settings", headers={"Authorization": "Bearer s3cret"}
    )
    assert r.status_code == 200


def test_admin_token_gate_covers_post_routes(client_with_admin_token):
    """POST /settings, /cache/purge, /cost/reset, /budget/reset all gated."""
    for path in ("/cache/purge", "/cost/reset", "/budget/reset"):
        r = client_with_admin_token.post(path)
        assert r.status_code == 401, f"{path} not gated"


def test_admin_token_gate_covers_stats_routes(client_with_admin_token):
    for path in ("/stats", "/stats/timeseries", "/stats/recent", "/metrics"):
        r = client_with_admin_token.get(path)
        assert r.status_code == 401, f"{path} not gated"


def test_healthz_stays_open_with_admin_token(client_with_admin_token):
    """/healthz is never gated — it's the operator-visible smoke check."""
    r = client_with_admin_token.get("/healthz")
    assert r.status_code == 200
    assert r.json()["admin_token_required"] is True


def test_healthz_reports_admin_token_state(client_no_admin_token):
    r = client_no_admin_token.get("/healthz")
    assert r.status_code == 200
    assert r.json()["admin_token_required"] is False
