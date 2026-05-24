"""Shared fixtures for the middleout-proxy test suite."""
from __future__ import annotations

from pathlib import Path

import pytest

from middleout_proxy.compression import PayloadCompressor
from middleout_proxy.config import BLOCKED_AUTH_ENV_VARS, Settings


@pytest.fixture(autouse=True)
def _scrub_blocked_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure no proxy-side auth env vars leak into tests.

    `load_settings()` refuses to start when any of `BLOCKED_AUTH_ENV_VARS` are set,
    so we strip them for every test to keep tests isolated from the user env.
    """
    for name in BLOCKED_AUTH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def default_settings() -> Settings:
    """Fresh `Settings()` instance with defaults."""
    return Settings()


@pytest.fixture
def tmp_audit_dir(tmp_path: Path) -> Path:
    """Per-test audit log directory inside pytest's `tmp_path`."""
    return tmp_path / "middleout-logs"


@pytest.fixture
def compressor(default_settings: Settings) -> PayloadCompressor:
    """Fresh `PayloadCompressor` bound to default settings."""
    return PayloadCompressor(default_settings)


@pytest.fixture
def mock_oauth_headers() -> dict[str, str]:
    """Headers dict containing a valid OAuth `Authorization: Bearer ...` header.

    Suitable for handing to `_forward_request_headers` (with a HeaderMap-style wrapper).
    """
    return {
        "authorization": "Bearer oauth-token-from-claude-code-login",
        "content-type": "application/json",
        "anthropic-beta": "claude-code-feature",
    }
