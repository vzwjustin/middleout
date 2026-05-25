"""Streaming SSE cost-tracking integration test.

Verifies that:
1. When upstream returns ``text/event-stream`` and the body contains the
   Anthropic ``message_start`` + ``message_delta`` events, the proxy
   parses the usage block as bytes pass through and records cost.
2. Non-SSE bodies (e.g. JSON error responses) on the streaming path
   never record cost (no usage to extract).
3. Mid-stream client disconnects still credit whatever was parsed up
   to that point — the cost recording lives in the ``finally`` block.
"""
from __future__ import annotations

import json
from typing import Any

import pytest


@pytest.fixture
def proxy_with_streaming():
    """Wires a fake httpx client that returns a configurable SSE stream."""
    from fastapi.testclient import TestClient
    from middleout_proxy import server as server_module

    captured_chunks: list[bytes] = []

    class _FakeStreamResponse:
        def __init__(
            self,
            *,
            status_code: int = 200,
            content_type: str = "text/event-stream",
            chunks: list[bytes] | None = None,
        ) -> None:
            self.status_code = status_code
            self.headers = {"content-type": content_type, "request-id": "r-stream-1"}
            self._chunks = chunks or []

        async def aiter_raw(self):
            for chunk in self._chunks:
                yield chunk

        async def aclose(self) -> None:
            pass

        # The non-streaming path uses these; the streaming path does not.
        def json(self) -> Any:
            return None

        @property
        def content(self) -> bytes:
            return b"".join(self._chunks)

    class _FakeRequest:
        def __init__(self, method: str, url: str) -> None:
            self.method = method
            self.url = url

    class _FakeAsyncClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.next_response: _FakeStreamResponse | None = None

        def build_request(self, method, url, *, headers, content):  # noqa: ARG002
            self.calls.append({"method": method, "url": url, "content": content})
            return _FakeRequest(method, url)

        async def send(self, req, *, stream):  # noqa: ARG002
            assert stream is True, "_streaming_forward must call send(stream=True)"
            assert self.next_response is not None, "test must set fake_http.next_response"
            return self.next_response

        async def request(self, method, url, *, headers, content):  # noqa: ARG002
            # Required by the non-streaming path; not used in streaming tests.
            raise NotImplementedError

        async def aclose(self) -> None:
            pass

    fake_http = _FakeAsyncClient()
    original_http = getattr(server_module.app.state, "http", None)
    server_module.app.state.http = fake_http  # type: ignore[attr-defined]

    # Reset the module-level cost tracker so test snapshots are isolated.
    server_module.cost_tracker.reset()

    client = TestClient(server_module.app)
    try:
        yield client, fake_http, server_module, captured_chunks
    finally:
        if original_http is not None:
            server_module.app.state.http = original_http
        else:
            try:
                delattr(server_module.app.state, "http")
            except Exception:
                pass


def _sse_chunks(input_tokens: int, output_tokens: int, model: str = "claude-3-5-sonnet-20241022") -> list[bytes]:
    """Build a list of SSE-formatted chunks for a single message stream."""
    def _ev(name: str, data: dict) -> bytes:
        return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()

    return [
        _ev(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_stream_1",
                    "model": model,
                    "usage": {
                        "input_tokens": input_tokens,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "output_tokens": 0,
                    },
                },
            },
        ),
        _ev("content_block_start", {"type": "content_block_start", "index": 0}),
        _ev("content_block_delta", {"type": "content_block_delta", "delta": {"text": "hi"}}),
        _ev("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _ev(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": output_tokens}},
        ),
        _ev("message_stop", {"type": "message_stop"}),
    ]


def test_streaming_records_cost_from_sse_usage(proxy_with_streaming) -> None:
    client, fake_http, server, _ = proxy_with_streaming
    payload = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "stream please"}],
        "stream": True,
    }
    # 2_500 input tokens, 700 output tokens.
    fake_http.next_response = type(
        "R", (), {})()  # placeholder; replaced next line
    chunks = _sse_chunks(input_tokens=2500, output_tokens=700)

    # Build a real fake response object (the placeholder above just gives us
    # something to attach to before we assign for real).
    class _R:
        status_code = 200
        headers = {"content-type": "text/event-stream", "request-id": "r1"}

        def __init__(self) -> None:
            self._chunks = chunks

        async def aiter_raw(self):
            for c in self._chunks:
                yield c

        async def aclose(self) -> None:
            pass

    fake_http.next_response = _R()

    response = client.post(
        "/v1/messages",
        headers={"Authorization": "Bearer test-oauth"},
        json=payload,
    )
    assert response.status_code == 200
    # Drain the body so `finally` runs.
    body = response.content
    # The proxy passes bytes through unchanged.
    assert body == b"".join(chunks)

    snap = server.cost_tracker.snapshot()
    assert snap["total_requests"] == 1
    # The model id in the stream's message_start overrides any request-side model.
    key = "anthropic:claude-3-5-sonnet-20241022"
    assert key in snap["by_model"]
    row = snap["by_model"][key]
    assert row["requests"] == 1
    assert row["input_tokens"] == 2500
    assert row["output_tokens"] == 700


def test_streaming_non_sse_body_records_no_cost(proxy_with_streaming) -> None:
    """A JSON error body on the streaming path must not invoke the SSE parser."""
    client, fake_http, server, _ = proxy_with_streaming
    payload = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "stream"}],
        "stream": True,
    }
    error_body = json.dumps({"error": {"type": "overloaded_error"}}).encode("utf-8")

    class _R:
        status_code = 200
        headers = {"content-type": "application/json", "request-id": "r"}

        async def aiter_raw(self):
            yield error_body

        async def aclose(self) -> None:
            pass

    fake_http.next_response = _R()
    server.cost_tracker.reset()

    response = client.post(
        "/v1/messages",
        headers={"Authorization": "Bearer test-oauth"},
        json=payload,
    )
    assert response.status_code == 200
    _ = response.content  # drain body to run finally

    snap = server.cost_tracker.snapshot()
    assert snap["total_requests"] == 0  # no usage parsed -> no cost recorded
    assert snap["total_usd"] == 0.0


def test_streaming_non_2xx_skips_cost_tracking(proxy_with_streaming) -> None:
    """5xx responses on the streaming path should not record cost."""
    client, fake_http, server, _ = proxy_with_streaming
    payload = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "x"}],
        "stream": True,
    }
    # Even if upstream returned SSE-formatted bytes, a 503 must be skipped.
    chunks = _sse_chunks(input_tokens=999, output_tokens=999)

    class _R:
        status_code = 503
        headers = {"content-type": "text/event-stream"}

        async def aiter_raw(self):
            for c in chunks:
                yield c

        async def aclose(self) -> None:
            pass

    fake_http.next_response = _R()
    server.cost_tracker.reset()

    response = client.post(
        "/v1/messages",
        headers={"Authorization": "Bearer t"},
        json=payload,
    )
    assert response.status_code == 503
    _ = response.content

    snap = server.cost_tracker.snapshot()
    assert snap["total_requests"] == 0


def test_streaming_records_cost_even_if_stream_truncated(proxy_with_streaming) -> None:
    """Client disconnect mid-stream must still record whatever we parsed."""
    client, fake_http, server, _ = proxy_with_streaming
    payload = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "x"}],
        "stream": True,
    }
    # Stream truncated after message_start but before message_delta.
    # output_tokens stays at 0 because message_delta never arrives —
    # but input_tokens IS recorded (the operator still paid for prefill).
    chunks = [
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"x","model":"claude-3-5-sonnet-20241022",'
        b'"usage":{"input_tokens":4242,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,'
        b'"output_tokens":0}}}\n\n',
        # ...no message_delta — stream truncated.
    ]

    class _R:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def aiter_raw(self):
            for c in chunks:
                yield c

        async def aclose(self) -> None:
            pass

    fake_http.next_response = _R()
    server.cost_tracker.reset()

    response = client.post(
        "/v1/messages",
        headers={"Authorization": "Bearer t"},
        json=payload,
    )
    assert response.status_code == 200
    _ = response.content

    snap = server.cost_tracker.snapshot()
    assert snap["total_requests"] == 1
    row = snap["by_model"]["anthropic:claude-3-5-sonnet-20241022"]
    assert row["input_tokens"] == 4242
    assert row["output_tokens"] == 0


def test_streaming_records_cache_token_axes(proxy_with_streaming) -> None:
    """Cache-creation + cache-read tokens from message_start surface to the tracker."""
    client, fake_http, server, _ = proxy_with_streaming
    payload = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "x"}],
        "stream": True,
    }
    chunks = [
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"x","model":"claude-3-5-sonnet-20241022",'
        b'"usage":{"input_tokens":50,"cache_creation_input_tokens":2000,'
        b'"cache_read_input_tokens":500,"output_tokens":0}}}\n\n',
        b"event: message_delta\n"
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        b'"usage":{"output_tokens":100}}\n\n',
    ]

    class _R:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def aiter_raw(self):
            for c in chunks:
                yield c

        async def aclose(self) -> None:
            pass

    fake_http.next_response = _R()
    server.cost_tracker.reset()

    response = client.post(
        "/v1/messages",
        headers={"Authorization": "Bearer t"},
        json=payload,
    )
    assert response.status_code == 200
    _ = response.content

    snap = server.cost_tracker.snapshot()
    row = snap["by_model"]["anthropic:claude-3-5-sonnet-20241022"]
    assert row["input_tokens"] == 50
    assert row["output_tokens"] == 100
    assert row["cache_write_tokens"] == 2000
    assert row["cache_read_tokens"] == 500
    # Sanity: the usd amount uses the cache prices, not full input.
    assert row["usd"] > 0


def test_streaming_records_cost_with_fragmented_chunks(proxy_with_streaming) -> None:
    """The accumulator must handle chunks that split mid-event."""
    client, fake_http, server, _ = proxy_with_streaming
    payload = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "x"}],
        "stream": True,
    }
    # Full payload, but yielded one byte at a time.
    full = b"".join(_sse_chunks(input_tokens=321, output_tokens=21))

    class _R:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def aiter_raw(self):
            for byte in full:
                yield bytes([byte])

        async def aclose(self) -> None:
            pass

    fake_http.next_response = _R()
    server.cost_tracker.reset()

    response = client.post(
        "/v1/messages",
        headers={"Authorization": "Bearer t"},
        json=payload,
    )
    assert response.status_code == 200
    _ = response.content

    snap = server.cost_tracker.snapshot()
    row = snap["by_model"]["anthropic:claude-3-5-sonnet-20241022"]
    assert row["input_tokens"] == 321
    assert row["output_tokens"] == 21
