"""Tests for server._is_streaming_messages — streaming-path detection."""
from __future__ import annotations

from middleout_proxy.server import _is_streaming_messages


def test_streaming_true_body_on_v1_messages():
    assert _is_streaming_messages("v1/messages", b'{"stream": true}') is True


def test_streaming_false_body_on_v1_messages():
    assert _is_streaming_messages("v1/messages", b'{"stream": false}') is False


def test_streaming_empty_body_is_not_streaming():
    assert _is_streaming_messages("v1/messages", b"") is False


def test_streaming_none_body_is_not_streaming():
    assert _is_streaming_messages("v1/messages", None) is False


def test_streaming_not_detected_on_count_tokens_path():
    # `count_tokens` is JSON but never streams.
    assert _is_streaming_messages("v1/messages/count_tokens", b'{"stream": true}') is False


def test_streaming_garbage_body_returns_false():
    assert _is_streaming_messages("v1/messages", b"not json at all") is False


def test_streaming_path_with_leading_and_trailing_slashes():
    # Path normalization should strip leading/trailing slashes.
    assert _is_streaming_messages("/v1/messages/", b'{"stream": true}') is True


def test_streaming_missing_stream_key_returns_false():
    # JSON valid but without "stream" key.
    assert _is_streaming_messages("v1/messages", b'{"model": "claude-3"}') is False


def test_streaming_other_paths_return_false():
    assert _is_streaming_messages("v1/models", b'{"stream": true}') is False
