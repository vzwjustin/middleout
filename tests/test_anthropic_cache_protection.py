"""Tests for Anthropic prompt-cache protection in PayloadCompressor."""
from __future__ import annotations

from middleout_proxy.compression import PayloadCompressor
from middleout_proxy.config import Settings


LONG = "L" * 5000  # Easily over max_text_chars below, will be compressed if not protected.
LONG2 = "M" * 5000
LONG3 = "N" * 5000
LONG4 = "O" * 5000
LONG5 = "P" * 5000


def _settings(**overrides) -> Settings:
    base = {
        "max_text_chars": 1000,
        "min_omission_chars": 200,
        "input_compression_enabled": True,
        "jl_dedupe_enabled": False,
        "caveman_enabled": False,
        "rtk_enabled": False,
        "compression_cache_enabled": False,
        "compress_system": True,
        "compress_tool_results": True,
        "preserve_anthropic_cache": True,
    }
    base.update(overrides)
    return Settings(**base)


def test_no_cache_control_means_everything_compressible():
    settings = _settings()
    payload = {
        "system": [
            {"type": "text", "text": LONG},
            {"type": "text", "text": LONG2},
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": LONG3},
                    {"type": "text", "text": LONG4},
                ],
            }
        ],
    }
    transformed, audit = PayloadCompressor(settings).compress_request_payload(
        payload, endpoint="v1/messages"
    )
    # All four long blocks should have been mutated and reported.
    assert transformed["system"][0]["text"] != LONG
    assert transformed["system"][1]["text"] != LONG2
    assert transformed["messages"][0]["content"][0]["text"] != LONG3
    assert transformed["messages"][0]["content"][1]["text"] != LONG4
    assert audit.protected_blocks == 0
    assert len(audit.events) >= 4


def test_cache_control_on_last_system_block_protects_all_system_and_no_messages():
    settings = _settings()
    payload = {
        "system": [
            {"type": "text", "text": LONG},
            {"type": "text", "text": LONG2, "cache_control": {"type": "ephemeral"}},
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": LONG3},
                    {"type": "text", "text": LONG4},
                ],
            }
        ],
    }
    transformed, audit = PayloadCompressor(settings).compress_request_payload(
        payload, endpoint="v1/messages"
    )
    # Both system blocks left untouched.
    assert transformed["system"][0]["text"] == LONG
    assert transformed["system"][1]["text"] == LONG2
    # Both message blocks compressed.
    assert transformed["messages"][0]["content"][0]["text"] != LONG3
    assert transformed["messages"][0]["content"][1]["text"] != LONG4
    # The 2 system blocks should be counted as protected.
    assert audit.protected_blocks == 2


def test_cache_control_mid_messages_partitions_protection():
    settings = _settings(compress_system=False)  # Focus the test on messages.
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": LONG},
                    {"type": "text", "text": LONG2},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": LONG3}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": LONG4},
                    {
                        "type": "text",
                        "text": LONG5,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": LONG},  # AFTER the marker.
                ],
            },
        ],
    }
    transformed, audit = PayloadCompressor(settings).compress_request_payload(
        payload, endpoint="v1/messages"
    )

    # messages[0] entirely protected.
    assert transformed["messages"][0]["content"][0]["text"] == LONG
    assert transformed["messages"][0]["content"][1]["text"] == LONG2
    # messages[1] entirely protected.
    assert transformed["messages"][1]["content"][0]["text"] == LONG3
    # messages[2]: blocks 0 and 1 protected.
    assert transformed["messages"][2]["content"][0]["text"] == LONG4
    assert transformed["messages"][2]["content"][1]["text"] == LONG5
    # messages[2]: block 2 is AFTER the marker -> must be compressed.
    assert transformed["messages"][2]["content"][2]["text"] != LONG

    # protected_blocks count: 2 + 1 + 2 = 5 blocks before/at the marker.
    assert audit.protected_blocks == 5


def test_preserve_anthropic_cache_false_disables_protection():
    settings = _settings(preserve_anthropic_cache=False)
    payload = {
        "system": [
            {"type": "text", "text": LONG, "cache_control": {"type": "ephemeral"}},
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": LONG2}],
            }
        ],
    }
    transformed, audit = PayloadCompressor(settings).compress_request_payload(
        payload, endpoint="v1/messages"
    )
    # Even the cache_control-marked block is compressed.
    assert transformed["system"][0]["text"] != LONG
    assert transformed["messages"][0]["content"][0]["text"] != LONG2
    assert audit.protected_blocks == 0


def test_bare_string_system_with_message_cache_marker_is_protected():
    settings = _settings()
    payload = {
        "system": LONG,  # bare string
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": LONG2,
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            }
        ],
    }
    transformed, audit = PayloadCompressor(settings).compress_request_payload(
        payload, endpoint="v1/messages"
    )
    # Bare-string system is protected because there exists a cache marker somewhere.
    assert transformed["system"] == LONG
    # The marked message block is also protected.
    assert transformed["messages"][0]["content"][0]["text"] == LONG2
    # protected_blocks counts: 1 (bare-string system) + 1 (message block) = 2.
    assert audit.protected_blocks == 2


def test_protected_blocks_counter_increments_via_compression_audit():
    settings = _settings()
    payload = {
        "system": [
            {
                "type": "text",
                "text": LONG,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        "messages": [{"role": "user", "content": [{"type": "text", "text": LONG2}]}],
    }
    _, audit = PayloadCompressor(settings).compress_request_payload(
        payload, endpoint="v1/messages"
    )
    # The audit object's protected_blocks field is incremented.
    assert audit.protected_blocks >= 1
    # And the dict serialization should round-trip the same number.
    assert audit.to_dict()["protected_blocks"] == audit.protected_blocks
