"""Edge-case tests for compression.py: PayloadCompressor & middle_out_text."""
from __future__ import annotations

from middleout_proxy.compression import PayloadCompressor, middle_out_text
from middleout_proxy.config import Settings


def _settings(**overrides) -> Settings:
    """Conservative defaults for these edge-case tests: no engines, no JL."""
    base = {
        "input_compression_enabled": True,
        "jl_dedupe_enabled": False,
        "caveman_enabled": False,
        "rtk_enabled": False,
        "compression_cache_enabled": False,
        "preserve_anthropic_cache": True,
    }
    base.update(overrides)
    return Settings(**base)


def test_empty_messages_list_produces_no_events():
    settings = _settings(max_text_chars=1000, min_omission_chars=200)
    transformed, audit = PayloadCompressor(settings).compress_request_payload(
        {"messages": []}, endpoint="v1/messages"
    )
    assert transformed == {"messages": []}
    assert audit.events == []
    assert audit.touched is False


def test_content_is_none_does_not_raise():
    settings = _settings(max_text_chars=1000, min_omission_chars=200)
    payload = {"messages": [{"role": "user", "content": None}]}
    transformed, audit = PayloadCompressor(settings).compress_request_payload(
        payload, endpoint="v1/messages"
    )
    # `None` content should pass through unchanged and not raise.
    assert transformed["messages"][0]["content"] is None
    assert audit.events == []


def test_bare_string_exactly_at_threshold_is_not_compressed():
    settings = _settings(max_text_chars=1000, min_omission_chars=200)
    text = "x" * 1000  # exactly at the cap
    payload = {"messages": [{"role": "user", "content": text}]}
    transformed, audit = PayloadCompressor(settings).compress_request_payload(
        payload, endpoint="v1/messages"
    )
    assert transformed["messages"][0]["content"] == text
    assert audit.events == []


def test_bare_string_well_over_threshold_is_compressed():
    settings = _settings(max_text_chars=1000, min_omission_chars=200)
    text = "x" * 5000
    payload = {"messages": [{"role": "user", "content": text}]}
    transformed, audit = PayloadCompressor(settings).compress_request_payload(
        payload, endpoint="v1/messages"
    )
    assert transformed["messages"][0]["content"] != text
    assert any(event.mode == "middle-out" for event in audit.events)
    assert audit.chars_saved > 0


def test_deeply_nested_tool_result_text_is_compressed():
    settings = _settings(
        max_text_chars=1000,
        min_omission_chars=200,
        compress_tool_results=True,
    )
    long_text = "y" * 5000
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "abc",
                        "content": [
                            {"type": "text", "text": long_text},
                        ],
                    }
                ],
            }
        ]
    }
    transformed, audit = PayloadCompressor(settings).compress_request_payload(
        payload, endpoint="v1/messages"
    )
    new_text = transformed["messages"][0]["content"][0]["content"][0]["text"]
    assert new_text != long_text
    assert "middle-out compressed locally" in new_text
    assert audit.chars_saved > 0


def test_input_compression_disabled_is_identity():
    settings = _settings(
        input_compression_enabled=False,
        max_text_chars=1000,
        min_omission_chars=200,
    )
    long_text = "z" * 5000
    payload = {"messages": [{"role": "user", "content": long_text}]}
    transformed, audit = PayloadCompressor(settings).compress_request_payload(
        payload, endpoint="v1/messages"
    )
    assert transformed == payload
    assert audit.events == []


def test_non_dict_block_in_content_list_is_passed_through():
    settings = _settings(max_text_chars=1000, min_omission_chars=200)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    "stray-bare-string",  # not a dict, must be skipped
                    {"type": "text", "text": "x" * 5000},
                ],
            }
        ]
    }
    transformed, audit = PayloadCompressor(settings).compress_request_payload(
        payload, endpoint="v1/messages"
    )
    # First entry untouched, second compressed.
    assert transformed["messages"][0]["content"][0] == "stray-bare-string"
    assert transformed["messages"][0]["content"][1]["text"] != "x" * 5000
    assert audit.chars_saved > 0


def test_head_fraction_005_boundary_works():
    text = "A" * 5000 + "MIDDLE" * 200 + "Z" * 5000
    out = middle_out_text(text, max_chars=2000, min_omission_chars=200, head_fraction=0.05)
    # Compressed output must be strictly shorter than the original.
    assert len(out) < len(text)
    # And must contain the marker.
    assert "middle-out compressed locally" in out
    # Head is tiny but the floor guarantees at least 64 chars.
    assert out.startswith("A" * 64)


def test_head_fraction_095_boundary_works():
    text = "A" * 5000 + "MIDDLE" * 200 + "Z" * 5000
    out = middle_out_text(text, max_chars=2000, min_omission_chars=200, head_fraction=0.95)
    assert len(out) < len(text)
    assert "middle-out compressed locally" in out
    # Tail floor is 64.
    assert out.endswith("Z" * 64)


def test_min_omission_chars_guard_prevents_short_text_compression():
    # max_chars=1000, min_omission=10000 means: only worth compressing if we'd drop >=10k chars.
    text = "x" * 1100  # only ~100 chars over the cap
    out = middle_out_text(text, max_chars=1000, min_omission_chars=10_000, head_fraction=0.55)
    assert out == text  # guard short-circuits the compression


def test_tool_result_with_compress_tool_results_false_is_not_compressed():
    settings = _settings(
        max_text_chars=1000,
        min_omission_chars=200,
        compress_tool_results=False,
    )
    long_text = "y" * 5000
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "abc",
                        "content": [{"type": "text", "text": long_text}],
                    }
                ],
            }
        ]
    }
    transformed, audit = PayloadCompressor(settings).compress_request_payload(
        payload, endpoint="v1/messages"
    )
    new_text = transformed["messages"][0]["content"][0]["content"][0]["text"]
    assert new_text == long_text
    assert audit.events == []


def test_system_as_list_with_compress_system_true_visits_each_block():
    settings = _settings(
        max_text_chars=1000,
        min_omission_chars=200,
        compress_system=True,
    )
    long_text = "s" * 5000
    payload = {
        "system": [
            {"type": "text", "text": long_text},
            {"type": "text", "text": long_text + "different-tail"},
        ],
        "messages": [{"role": "user", "content": "hello"}],
    }
    transformed, audit = PayloadCompressor(settings).compress_request_payload(
        payload, endpoint="v1/messages"
    )
    # Both system blocks should have been compressed.
    new_first = transformed["system"][0]["text"]
    new_second = transformed["system"][1]["text"]
    assert new_first != long_text
    assert new_second != (long_text + "different-tail")
    # Two text-bearing blocks; both should generate events.
    text_events = [e for e in audit.events if e.path.startswith("system")]
    assert len(text_events) >= 2
