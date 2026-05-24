"""Unit tests for the provider adapter scaffolding."""

from __future__ import annotations

import pytest

from middleout_proxy.providers import (
    REGISTRY,
    AdapterError,
    AdapterNotImplemented,
    RequestIR,
    select_adapter,
)
from middleout_proxy.providers.anthropic import AnthropicAdapter
from middleout_proxy.providers.registry import routes_snapshot


# -- RequestIR --------------------------------------------------------------


def test_request_ir_model_from_payload() -> None:
    ir = RequestIR(payload={"model": "claude-3-5-sonnet"})
    assert ir.model == "claude-3-5-sonnet"
    assert ir.effective_model == "claude-3-5-sonnet"


def test_request_ir_hint_overrides_payload_model() -> None:
    ir = RequestIR(payload={"model": "claude-3-5-sonnet"}, model_hint="gpt-4o")
    assert ir.model == "claude-3-5-sonnet"
    assert ir.effective_model == "gpt-4o"


def test_request_ir_missing_model_returns_empty() -> None:
    ir = RequestIR(payload={})
    assert ir.model == ""
    assert ir.effective_model == ""


# -- Registry ---------------------------------------------------------------


def test_registry_includes_all_scaffolded_adapters() -> None:
    assert "anthropic" in REGISTRY
    assert "openai" in REGISTRY
    assert "google" in REGISTRY
    assert "ollama" in REGISTRY


def test_select_adapter_by_direct_name_hint() -> None:
    adapter = select_adapter(model="", model_hint="openai")
    assert adapter.name == "openai"


def test_select_adapter_by_model_glob_from_body() -> None:
    adapter = select_adapter(model="claude-3-5-sonnet-20240620")
    assert adapter.name == "anthropic"


def test_select_adapter_by_openai_pattern() -> None:
    adapter = select_adapter(model="gpt-4o-mini")
    assert adapter.name == "openai"


def test_select_adapter_by_gemini_pattern() -> None:
    adapter = select_adapter(model="gemini-1.5-flash")
    assert adapter.name == "google"


def test_select_adapter_by_ollama_pattern() -> None:
    adapter = select_adapter(model="ollama:llama3.1:70b")
    assert adapter.name == "ollama"


def test_select_adapter_unknown_model_falls_back_to_anthropic() -> None:
    adapter = select_adapter(model="totally-unknown-model")
    assert adapter.name == "anthropic"


def test_select_adapter_hint_overrides_body_model() -> None:
    """Explicit hint must win over the body model glob match."""
    adapter = select_adapter(model="gpt-4o", model_hint="anthropic")
    assert adapter.name == "anthropic"


def test_routes_snapshot_lists_globs() -> None:
    snap = routes_snapshot()
    assert "adapters" in snap
    assert "routes" in snap
    globs = {r["glob"] for r in snap["routes"]}
    assert "gpt-*" in globs
    assert "claude-*" in globs


# -- Anthropic identity adapter ---------------------------------------------


def test_anthropic_translate_request_is_identity() -> None:
    adapter = AnthropicAdapter()
    payload = {"model": "claude-3-5-sonnet", "messages": [{"role": "user", "content": "hi"}]}
    ir = RequestIR(payload=payload, headers={"authorization": "Bearer t"}, endpoint="v1/messages")
    url_path, headers, body = adapter.translate_request(ir)
    assert url_path == "v1/messages"
    assert headers == {"authorization": "Bearer t"}
    import json
    decoded = json.loads(body.decode("utf-8"))
    assert decoded == payload


def test_anthropic_translate_request_drops_x_brain_headers() -> None:
    """X-Brain-* headers are proxy-internal and must never reach Anthropic."""
    adapter = AnthropicAdapter()
    ir = RequestIR(
        payload={"model": "claude-3-5-sonnet"},
        headers={
            "authorization": "Bearer t",
            "x-brain-model-hint": "anthropic",
            "x-brain-replay": "1",
        },
        endpoint="v1/messages",
    )
    _, headers, _ = adapter.translate_request(ir)
    assert "authorization" in headers
    assert "x-brain-model-hint" not in headers
    assert "x-brain-replay" not in headers


def test_anthropic_translate_response_returns_ir() -> None:
    adapter = AnthropicAdapter()
    body = b'{"id":"msg_1","model":"claude-3-5-sonnet","content":[]}'
    ir = adapter.translate_response(
        status_code=200,
        headers={"content-type": "application/json"},
        body_bytes=body,
        media_type="application/json",
    )
    assert ir.status_code == 200
    assert ir.body_bytes == body
    assert isinstance(ir.payload, dict)
    assert ir.payload["id"] == "msg_1"


def test_anthropic_translate_response_tolerates_non_json() -> None:
    adapter = AnthropicAdapter()
    ir = adapter.translate_response(
        status_code=500,
        headers={"content-type": "text/plain"},
        body_bytes=b"Internal Server Error",
        media_type="text/plain",
    )
    assert ir.status_code == 500
    assert ir.payload is None


def test_anthropic_cost_provider() -> None:
    assert AnthropicAdapter().cost_provider() == "anthropic"


# -- OpenAI / Gemini / Ollama scaffolds raise AdapterNotImplemented --------


@pytest.mark.parametrize("name", ["openai", "google", "ollama"])
def test_non_anthropic_adapters_raise_not_implemented(name: str) -> None:
    adapter = REGISTRY[name]
    ir = RequestIR(payload={"model": "anything"})
    with pytest.raises(AdapterNotImplemented):
        adapter.translate_request(ir)
    with pytest.raises(AdapterNotImplemented):
        adapter.translate_response(
            status_code=200,
            headers={},
            body_bytes=b"",
            media_type="application/json",
        )


def test_adapter_not_implemented_is_adapter_error() -> None:
    """AdapterNotImplemented subclasses AdapterError so callers can catch broadly."""
    assert issubclass(AdapterNotImplemented, AdapterError)
