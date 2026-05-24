import pytest

from middleout_proxy.config import BLOCKED_AUTH_ENV_VARS, Settings, load_settings
from middleout_proxy.server import StrictSubscriptionAuthError, _forward_request_headers


class HeaderMap(dict):
    def get(self, key, default=None):
        return super().get(key.lower(), default)

    def items(self):
        return super().items()


def test_rejects_x_api_key_header():
    headers = HeaderMap(
        {
            "authorization": "Bearer oauth-token-from-claude-code-login",
            "x-api-key": "api-key-should-not-pass",
        }
    )
    with pytest.raises(StrictSubscriptionAuthError, match="rejected an X-Api-Key"):
        _forward_request_headers(headers, Settings())


def test_rejects_missing_oauth_authorization_header():
    headers = HeaderMap({"anthropic-beta": "claude-code-feature"})
    with pytest.raises(StrictSubscriptionAuthError, match="Missing OAuth Authorization"):
        _forward_request_headers(headers, Settings())


def test_allows_subscription_style_bearer_and_does_not_inject_key():
    headers = HeaderMap(
        {
            "authorization": "Bearer oauth-token-from-claude-code-login",
            "anthropic-beta": "claude-code-feature",
            "content-type": "application/json",
            "accept-encoding": "gzip",
        }
    )
    forwarded = _forward_request_headers(headers, Settings())
    assert forwarded["authorization"] == "Bearer oauth-token-from-claude-code-login"
    assert forwarded["anthropic-beta"] == "claude-code-feature"
    assert forwarded["content-type"] == "application/json"
    assert forwarded["anthropic-version"] == "2023-06-01"
    assert "x-api-key" not in forwarded
    assert "accept-encoding" not in forwarded


@pytest.mark.parametrize("env_name", BLOCKED_AUTH_ENV_VARS)
def test_proxy_refuses_to_start_with_api_key_or_custom_token_env(monkeypatch, env_name):
    for name in BLOCKED_AUTH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(env_name, "secret")

    with pytest.raises(ValueError, match="Strict subscription-only mode refuses"):
        load_settings()
