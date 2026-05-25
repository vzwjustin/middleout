"""Budget enforcement.

Default deployment is observe-only — limits inform `/budget` but never
reject requests. When `settings.budget_enforce` is set, requests above the
char or token limit are rejected with 429.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from middleout_proxy.budget import UsageBudget


class _FakeUpstream:
    """Drop-in stand-in for app.state.http that returns canned JSON."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def request(self, method, url, *, headers, content):  # noqa: ARG002
        self.calls.append({"method": method, "url": url})

        class _R:
            status_code = 200
            content = (
                b'{"id":"m1","model":"claude-3-5-sonnet",'
                b'"usage":{"input_tokens":1,"output_tokens":1},'
                b'"content":[{"type":"text","text":"ok"}]}'
            )
            headers = {"content-type": "application/json"}

            def json(self_inner):  # noqa: D401
                return json.loads(self_inner.content.decode("utf-8"))

        return _R()

    async def aclose(self):
        pass


def _set_enforce(monkeypatch, server_module, enforce: bool) -> None:
    import dataclasses
    new_settings = dataclasses.replace(server_module.settings, budget_enforce=enforce)
    monkeypatch.setattr(server_module, "settings", new_settings)


@pytest.fixture
def proxy(monkeypatch):
    from middleout_proxy import server as server_module

    _set_enforce(monkeypatch, server_module, False)
    server_module.usage_budget = UsageBudget(char_limit=None, token_limit=None)
    with TestClient(server_module.app) as client:
        # The lifespan startup overwrites `app.state.http` with a real
        # httpx.AsyncClient; we need to install the fake AFTER entering.
        server_module.app.state.http = _FakeUpstream()
        yield client, server_module


def _send(client) -> int:
    r = client.post(
        "/v1/messages",
        headers={"Authorization": "Bearer t"},
        json={"model": "claude-3-5-sonnet", "messages": [{"role": "user", "content": "hi"}]},
    )
    return r.status_code


def test_observe_only_does_not_reject(proxy, monkeypatch):
    client, server_module = proxy
    # Limit set, enforcement off → still 200.
    _set_enforce(monkeypatch, server_module, False)
    server_module.usage_budget = UsageBudget(char_limit=1, token_limit=None)
    server_module.usage_budget.record(chars=100, tokens=0)
    assert _send(client) == 200


def test_enforce_below_limit_passes(proxy, monkeypatch):
    client, server_module = proxy
    _set_enforce(monkeypatch, server_module, True)
    server_module.usage_budget = UsageBudget(char_limit=1_000_000, token_limit=None)
    assert _send(client) == 200


def test_enforce_above_char_limit_rejects(proxy, monkeypatch):
    client, server_module = proxy
    _set_enforce(monkeypatch, server_module, True)
    server_module.usage_budget = UsageBudget(char_limit=10, token_limit=None)
    server_module.usage_budget.record(chars=100, tokens=0)
    r = client.post(
        "/v1/messages",
        headers={"Authorization": "Bearer t"},
        json={"model": "claude-3-5-sonnet", "messages": []},
    )
    assert r.status_code == 429
    body = r.json()
    assert body["error"]["type"] == "budget_exceeded_error"
    assert r.headers.get("x-brain-budget") == "exceeded"


def test_enforce_above_token_limit_rejects(proxy, monkeypatch):
    client, server_module = proxy
    _set_enforce(monkeypatch, server_module, True)
    server_module.usage_budget = UsageBudget(char_limit=None, token_limit=10)
    server_module.usage_budget.record(chars=0, tokens=100)
    assert _send(client) == 429


def test_reset_clears_enforcement(proxy, monkeypatch):
    client, server_module = proxy
    _set_enforce(monkeypatch, server_module, True)
    server_module.usage_budget = UsageBudget(char_limit=10, token_limit=None)
    server_module.usage_budget.record(chars=100, tokens=0)
    assert _send(client) == 429
    server_module.usage_budget.reset()
    assert _send(client) == 200
