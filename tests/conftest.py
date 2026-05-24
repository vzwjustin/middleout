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


@pytest.fixture(autouse=True)
def _reset_server_runtime() -> "object":
    """Snapshot+restore server._runtime around every test.

    Without this, a test that flips a runtime flag via POST /settings
    (e.g. enabling l1_cache or l2_cache) leaks state into the next test
    and causes order-dependent flakes. The fixture is best-effort: if
    server.py hasn't been imported yet we no-op.
    """
    try:
        from middleout_proxy import server as _srv
    except Exception:
        yield
        return
    snapshot = dict(_srv._runtime)
    # Snapshot the L2 enabled bit too -- the POST /settings handler mirrors
    # the runtime flag onto l2_cache.enabled directly.
    l2_enabled_snapshot: bool | None = None
    try:
        l2_enabled_snapshot = bool(_srv.l2_cache.enabled)
    except Exception:
        l2_enabled_snapshot = None
    try:
        yield
    finally:
        _srv._runtime.clear()
        _srv._runtime.update(snapshot)
        if l2_enabled_snapshot is not None:
            try:
                _srv.l2_cache.enabled = l2_enabled_snapshot
            except Exception:
                pass


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
