"""Adversarial auth tests — locks down the strict-subscription invariants.

These tests assume the codebase has applied the auth-fortress hardening:
- BLOCKED_AUTH_ENV_VARS uses `is not None` (rejects empty strings)
- BLOCKED_AUTH_ENV_VARS uppercased + case-insensitive comparison
- _forward_response_headers strips auth-leaking headers
- _forward_request_headers rejects comma-folded Authorization
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from middleout_proxy.config import BLOCKED_AUTH_ENV_VARS, load_settings


# -- BLOCKED_AUTH_ENV_VARS exhaustiveness -------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "ANTHROPIC_BEARER_TOKEN",
        "CLAUDE_CODE_API_KEY_HELPER",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "AWS_BEDROCK_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "VERTEX_API_KEY",
    ],
)
def test_blocked_list_contains_expanded_entries(name: str) -> None:
    assert name in BLOCKED_AUTH_ENV_VARS, (
        f"{name} should be in BLOCKED_AUTH_ENV_VARS to prevent API-key bypass"
    )


@pytest.mark.parametrize("name", BLOCKED_AUTH_ENV_VARS)
def test_empty_string_env_var_still_blocked(monkeypatch, name: str) -> None:
    """Empty-string env var is a non-None value and must block startup."""
    monkeypatch.setenv(name, "")
    with pytest.raises(ValueError, match="refuses proxy-side auth"):
        load_settings()


@pytest.mark.parametrize("name", BLOCKED_AUTH_ENV_VARS)
def test_set_env_var_blocked(monkeypatch, name: str) -> None:
    monkeypatch.setenv(name, "should-not-pass")
    with pytest.raises(ValueError, match="refuses proxy-side auth"):
        load_settings()


def test_lowercase_env_var_name_also_blocked(monkeypatch) -> None:
    """POSIX env vars are case-sensitive but Claude Code wrappers may
    accidentally set lowercase variants. The case-insensitive comparison
    in load_settings catches them."""
    # Skip if we're on a non-POSIX system where case is preserved differently.
    monkeypatch.setenv("anthropic_api_key", "secret")
    with pytest.raises(ValueError):
        load_settings()


# -- comma-folded Authorization -----------------------------------------------


def test_comma_folded_authorization_rejected() -> None:
    from middleout_proxy.server import (
        StrictSubscriptionAuthError,
        _forward_request_headers,
        settings,
    )
    headers = {"authorization": "Bearer goodtoken, ApiKey sk-ant-abc"}
    with pytest.raises(StrictSubscriptionAuthError, match="comma"):
        _forward_request_headers(headers, settings)


def test_clean_bearer_still_accepted() -> None:
    from middleout_proxy.server import (
        _forward_request_headers,
        settings,
    )
    headers = {"authorization": "Bearer oauth-token-no-commas-here"}
    forwarded = _forward_request_headers(headers, settings)
    assert forwarded["authorization"] == "Bearer oauth-token-no-commas-here"


# -- response-side header stripping -------------------------------------------


@pytest.mark.parametrize(
    "leaky_header",
    ["authorization", "Authorization", "x-api-key", "X-Api-Key", "anthropic-api-key",
     "proxy-authorization", "set-cookie", "Set-Cookie"],
)
def test_response_headers_strip_auth_leakage(leaky_header: str) -> None:
    from middleout_proxy.server import _forward_response_headers
    headers = {
        "content-type": "application/json",
        leaky_header: "should-never-leak",
        "x-request-id": "req-123",
    }
    out = _forward_response_headers(headers)
    assert "x-request-id" in out
    assert "content-type" in out
    # Strip is case-insensitive — every casing of the leaky header must be gone.
    for key in out:
        assert leaky_header.lower() != key.lower(), f"{leaky_header} leaked through"


# -- mixed-case x-api-key rejection -------------------------------------------


@pytest.mark.parametrize(
    "header_name",
    ["X-Api-Key", "x-api-key", "X-API-KEY", "x-Api-Key"],
)
def test_mixed_case_x_api_key_rejected(header_name: str) -> None:
    from middleout_proxy.server import (
        StrictSubscriptionAuthError,
        _forward_request_headers,
        settings,
    )
    headers = {"authorization": "Bearer valid", header_name: "sk-ant-bad"}
    with pytest.raises(StrictSubscriptionAuthError, match="X-Api-Key"):
        _forward_request_headers(headers, settings)


@pytest.mark.parametrize(
    "header_name",
    ["Anthropic-Api-Key", "anthropic-api-key", "ANTHROPIC-API-KEY"],
)
def test_mixed_case_anthropic_api_key_rejected(header_name: str) -> None:
    from middleout_proxy.server import (
        StrictSubscriptionAuthError,
        _forward_request_headers,
        settings,
    )
    headers = {"authorization": "Bearer valid", header_name: "sk-ant-bad"}
    with pytest.raises(StrictSubscriptionAuthError):
        _forward_request_headers(headers, settings)


# -- /healthz invariants ------------------------------------------------------


def test_healthz_advertises_no_api_key_support() -> None:
    from middleout_proxy.server import app
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["api_keys_supported"] is False
    assert data["api_key_headers_rejected"] is True
    assert data["api_key_injection"] is False


# -- /settings cannot disable auth --------------------------------------------


def test_settings_post_cannot_change_auth_mode() -> None:
    from middleout_proxy.server import app
    client = TestClient(app)
    # Attempt to inject auth-related keys into /settings POST — they must be ignored.
    client.post("/settings", json={"auth_mode": "api_key", "api_keys_supported": True})
    resp = client.get("/healthz")
    data = resp.json()
    assert data["api_keys_supported"] is False
    assert data["api_key_headers_rejected"] is True
    assert data["api_key_injection"] is False


def test_x_middleout_proxy_header_is_static() -> None:
    """The proxy identifier header must not contain user/host/PID info."""
    from middleout_proxy.server import (
        _forward_request_headers,
        settings,
    )
    headers = {"authorization": "Bearer x"}
    forwarded = _forward_request_headers(headers, settings)
    value = forwarded["x-middleout-proxy"]
    # Allow-list match: must be the documented static string.
    assert value == "middleout-claude-proxy/0.2.0-strict-subscription"


# -- audit content invariants -------------------------------------------------


def test_audit_log_never_contains_authorization(tmp_path, monkeypatch) -> None:
    """Drive a request through AuditLogger.record and confirm the resulting
    JSONL never contains the Authorization header or bearer token."""
    from middleout_proxy.audit import AuditLogger
    from middleout_proxy.compression import CompressionAudit
    from middleout_proxy.config import Settings

    log_dir = tmp_path / "logs"
    settings = Settings(audit_enabled=True, audit_log_dir=log_dir)
    logger = AuditLogger(settings)
    audit = CompressionAudit(endpoint="v1/messages")
    logger.record(
        method="POST",
        path="v1/messages",
        status_code=200,
        request_audit=audit,
        request_id="req-test-123",
    )

    log_file = log_dir / "audit.jsonl"
    assert log_file.exists()
    content = log_file.read_text()
    assert "Bearer" not in content
    assert "authorization" not in content.lower()
    assert "x-api-key" not in content.lower()
