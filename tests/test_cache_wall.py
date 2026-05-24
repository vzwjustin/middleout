"""Cache wall invariant tests.

These tests lock down the proxy's most important contract: bytes left of the
last cache_control marker MUST NOT change. Every other engine in the project
consults `CacheWall.is_protected` to decide what's touchable; if these tests
break, so does upstream prompt caching, and the proxy stops paying for itself.
"""

from __future__ import annotations

import copy
import json

import pytest

from middleout_proxy.cache_wall import (
    CacheWall,
    CacheWallViolation,
    WallMarker,
    assert_prefix_unchanged,
    compute_wall,
    iter_volatile_blocks,
)
from middleout_proxy.compression import (
    _is_block_protected,
    _payload_cache_protection,
)


def _ephemeral() -> dict:
    return {"type": "ephemeral"}


def _text(s: str, **extra) -> dict:
    return {"type": "text", "text": s, **extra}


# -- detection -----------------------------------------------------------------


def test_no_marker_yields_no_wall() -> None:
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    wall = compute_wall(payload)
    assert wall.marker is None
    assert not wall.has_marker
    assert wall.all_markers == ()
    assert not wall.is_protected(kind="message", msg_idx=0, block_idx=0)


def test_marker_on_system_block() -> None:
    payload = {
        "system": [_text("sys", cache_control=_ephemeral())],
        "messages": [{"role": "user", "content": "hi"}],
    }
    wall = compute_wall(payload)
    assert wall.marker == WallMarker("system", None, 0)
    assert wall.is_protected(kind="system", msg_idx=None, block_idx=0)
    assert not wall.is_protected(kind="message", msg_idx=0, block_idx=0)


def test_marker_on_tools() -> None:
    payload = {
        "tools": [{"name": "search", "cache_control": _ephemeral()}],
        "messages": [{"role": "user", "content": [_text("hi")]}],
    }
    wall = compute_wall(payload)
    assert wall.marker == WallMarker("tools", None, 0)
    assert wall.is_protected(kind="tools", msg_idx=None, block_idx=0)
    assert wall.is_protected(kind="system", msg_idx=None, block_idx=0)
    assert not wall.is_protected(kind="message", msg_idx=0, block_idx=0)


def test_marker_on_message_block() -> None:
    payload = {
        "messages": [
            {"role": "user", "content": [_text("hi", cache_control=_ephemeral())]},
            {"role": "assistant", "content": [_text("hey")]},
            {"role": "user", "content": [_text("more")]},
        ],
    }
    wall = compute_wall(payload)
    assert wall.marker == WallMarker("message", 0, 0)
    assert wall.is_protected(kind="message", msg_idx=0, block_idx=0)
    assert not wall.is_protected(kind="message", msg_idx=1, block_idx=0)


def test_picks_the_last_marker_in_processing_order() -> None:
    payload = {
        "system": [_text("sys", cache_control=_ephemeral())],
        "tools": [{"name": "t", "cache_control": _ephemeral()}],
        "messages": [
            {"role": "user", "content": [_text("a", cache_control=_ephemeral())]},
            {"role": "assistant", "content": [_text("b")]},
        ],
    }
    wall = compute_wall(payload)
    # Last in processing order is the message marker.
    assert wall.marker == WallMarker("message", 0, 0)
    assert len(wall.all_markers) == 3
    assert wall.is_protected(kind="system", msg_idx=None, block_idx=0)
    assert wall.is_protected(kind="tools", msg_idx=None, block_idx=0)
    assert wall.is_protected(kind="message", msg_idx=0, block_idx=0)
    assert not wall.is_protected(kind="message", msg_idx=1, block_idx=0)


def test_picks_last_when_markers_only_in_system_or_tools() -> None:
    payload = {
        "system": [_text("sys", cache_control=_ephemeral())],
        "tools": [{"name": "t", "cache_control": _ephemeral()}],
        "messages": [{"role": "user", "content": [_text("hi")]}],
    }
    wall = compute_wall(payload)
    # tools comes after system in processing order.
    assert wall.marker == WallMarker("tools", None, 0)


# -- protection oracle ---------------------------------------------------------


@pytest.mark.parametrize(
    "wall_marker, query, expected",
    [
        (WallMarker("system", None, 2), ("system", None, 0), True),
        (WallMarker("system", None, 2), ("system", None, 2), True),
        (WallMarker("system", None, 2), ("system", None, 3), False),
        (WallMarker("system", None, 0), ("tools", None, 0), False),
        (WallMarker("tools", None, 0), ("system", None, 5), True),
        (WallMarker("tools", None, 1), ("tools", None, 0), True),
        (WallMarker("tools", None, 1), ("tools", None, 2), False),
        (WallMarker("tools", None, 0), ("message", 0, 0), False),
        (WallMarker("message", 2, 1), ("system", None, 99), True),
        (WallMarker("message", 2, 1), ("tools", None, 99), True),
        (WallMarker("message", 2, 1), ("message", 1, 99), True),
        (WallMarker("message", 2, 1), ("message", 2, 0), True),
        (WallMarker("message", 2, 1), ("message", 2, 1), True),
        (WallMarker("message", 2, 1), ("message", 2, 2), False),
        (WallMarker("message", 2, 1), ("message", 3, 0), False),
    ],
)
def test_is_protected_matrix(wall_marker, query, expected) -> None:
    wall = CacheWall(marker=wall_marker)
    kind, msg_idx, block_idx = query
    assert wall.is_protected(kind=kind, msg_idx=msg_idx, block_idx=block_idx) is expected


def test_unknown_kind_is_not_protected() -> None:
    wall = CacheWall(marker=WallMarker("system", None, 0))
    assert not wall.is_protected(kind="not-a-real-kind", msg_idx=None, block_idx=0)


# -- auto-insert ---------------------------------------------------------------


def test_auto_insert_prefers_last_tool() -> None:
    payload = {
        "system": [_text("sys")],
        "tools": [{"name": "a"}, {"name": "b"}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    wall = compute_wall(payload, auto_insert=True)
    assert wall.auto_inserted is True
    assert wall.marker == WallMarker("tools", None, 1)
    assert payload["tools"][1]["cache_control"] == _ephemeral()
    assert "cache_control" not in payload["tools"][0]


def test_auto_insert_falls_back_to_last_system_block_when_no_tools() -> None:
    payload = {
        "system": [_text("a"), _text("b")],
        "messages": [{"role": "user", "content": "hi"}],
    }
    wall = compute_wall(payload, auto_insert=True)
    assert wall.auto_inserted is True
    assert wall.marker == WallMarker("system", None, 1)
    assert payload["system"][1]["cache_control"] == _ephemeral()
    assert "cache_control" not in payload["system"][0]


def test_auto_insert_promotes_string_system_to_list() -> None:
    payload = {
        "system": "you are a helpful assistant",
        "messages": [{"role": "user", "content": "hi"}],
    }
    wall = compute_wall(payload, auto_insert=True)
    assert wall.auto_inserted is True
    assert wall.marker == WallMarker("system", None, 0)
    assert isinstance(payload["system"], list)
    assert payload["system"][0]["text"] == "you are a helpful assistant"
    assert payload["system"][0]["cache_control"] == _ephemeral()


def test_auto_insert_no_anchor_yields_no_wall() -> None:
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    before = copy.deepcopy(payload)
    wall = compute_wall(payload, auto_insert=True)
    assert wall.marker is None
    assert wall.auto_inserted is False
    assert payload == before  # Unmodified.


def test_auto_insert_is_a_noop_when_marker_already_exists() -> None:
    payload = {
        "system": [_text("sys", cache_control=_ephemeral())],
        "tools": [{"name": "a"}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    before = copy.deepcopy(payload)
    wall = compute_wall(payload, auto_insert=True)
    assert wall.auto_inserted is False  # We did not insert; user already had one.
    assert wall.marker == WallMarker("system", None, 0)
    assert payload == before


# -- iter_volatile_blocks ------------------------------------------------------


def test_iter_volatile_blocks_skips_protected_only() -> None:
    payload = {
        "system": [_text("sys", cache_control=_ephemeral())],
        "messages": [
            {"role": "user", "content": [_text("a"), _text("b")]},
            {"role": "assistant", "content": [_text("c")]},
        ],
    }
    wall = compute_wall(payload)
    out = list(iter_volatile_blocks(payload, wall))
    # System block protected (wall is on system[0]). All message blocks are
    # volatile because the wall does not cross into messages.
    kinds = [k for (k, _, _, _) in out]
    msg_block_pairs = [(mi, bi) for (k, mi, bi, _) in out if k == "message"]
    assert kinds == ["message", "message", "message"]
    assert msg_block_pairs == [(0, 0), (0, 1), (1, 0)]


def test_iter_volatile_skips_messages_before_wall() -> None:
    payload = {
        "messages": [
            {"role": "user", "content": [_text("a"), _text("b", cache_control=_ephemeral())]},
            {"role": "assistant", "content": [_text("c")]},
            {"role": "user", "content": [_text("d")]},
        ],
    }
    wall = compute_wall(payload)
    out = list(iter_volatile_blocks(payload, wall))
    assert [(mi, bi) for (_, mi, bi, _) in out] == [(1, 0), (2, 0)]


def test_iter_volatile_handles_string_content() -> None:
    payload = {"messages": [{"role": "user", "content": "hello world"}]}
    wall = compute_wall(payload)
    out = list(iter_volatile_blocks(payload, wall))
    assert len(out) == 1
    kind, mi, bi, block = out[0]
    assert (kind, mi, bi, block) == ("message", 0, 0, "hello world")


# -- prefix-byte invariant -----------------------------------------------------


def test_assert_prefix_unchanged_passes_when_bytes_match() -> None:
    a = b'{"system":[{"type":"text","text":"sys","cache_control":{"type":"ephemeral"}}]}'
    b = a + b" trailing differs"
    wall = CacheWall(marker=WallMarker("system", None, 0))
    # No exception: prefix is identical for the first min(len(a), len(b)) bytes,
    # and our test caller specifies prefix_len.
    assert_prefix_unchanged(a, b, wall, prefix_len=len(a))


def test_assert_prefix_unchanged_raises_on_mutation() -> None:
    a = b'{"system":[{"type":"text","text":"sys"}]}'
    b = b'{"system":[{"type":"text","text":"SYS"}]}'  # one byte differs
    wall = CacheWall(marker=WallMarker("system", None, 0))
    with pytest.raises(CacheWallViolation) as excinfo:
        assert_prefix_unchanged(a, b, wall, prefix_len=len(a))
    assert "mutated at byte" in str(excinfo.value)


def test_assert_prefix_unchanged_is_noop_without_marker() -> None:
    a = b'{"a":1}'
    b = b'{"b":2}'
    wall = CacheWall(marker=None)
    # No marker = nothing protected, no exception even though bytes differ.
    assert_prefix_unchanged(a, b, wall)


# -- parity with legacy --------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        # No markers.
        {"messages": [{"role": "user", "content": "hi"}]},
        # System marker only.
        {
            "system": [_text("a", cache_control=_ephemeral())],
            "messages": [{"role": "user", "content": [_text("hi")]}],
        },
        # Message marker only.
        {
            "messages": [
                {"role": "user", "content": [_text("a", cache_control=_ephemeral())]},
                {"role": "assistant", "content": [_text("b")]},
            ],
        },
        # Multiple markers across system + messages.
        {
            "system": [_text("a", cache_control=_ephemeral())],
            "messages": [
                {"role": "user", "content": [_text("u1")]},
                {
                    "role": "assistant",
                    "content": [_text("a1"), _text("a2", cache_control=_ephemeral())],
                },
                {"role": "user", "content": [_text("u2")]},
            ],
        },
    ],
)
def test_parity_with_legacy_protection_on_messages(payload) -> None:
    """The new CacheWall must agree with the legacy oracle on every message
    block in a battery of representative payloads. The legacy oracle does not
    know about `tools`, so we restrict this comparison to system + message."""
    legacy = _payload_cache_protection(payload)
    wall = compute_wall(payload)

    # Compare on system blocks.
    sys_blocks = payload.get("system") or []
    if isinstance(sys_blocks, list):
        for i in range(len(sys_blocks)):
            new = wall.is_protected(kind="system", msg_idx=None, block_idx=i)
            old = _is_block_protected(legacy, kind="system", msg_idx=None, block_idx=i)
            assert new == old, f"system[{i}]: new={new}, legacy={old}"

    # Compare on message blocks.
    for mi, message in enumerate(payload.get("messages", [])):
        content = message.get("content")
        if isinstance(content, list):
            for bi in range(len(content)):
                new = wall.is_protected(kind="message", msg_idx=mi, block_idx=bi)
                old = _is_block_protected(legacy, kind="message", msg_idx=mi, block_idx=bi)
                assert new == old, f"messages[{mi}].content[{bi}]: new={new}, legacy={old}"


def test_json_round_trip_preserves_prefix_bytes() -> None:
    """Round-tripping a payload through json.dumps with stable separators
    must keep the cache-stable prefix identical."""
    payload = {
        "system": [_text("sys", cache_control=_ephemeral())],
        "messages": [{"role": "user", "content": [_text("hello")]}],
    }
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    decoded = json.loads(encoded.decode("utf-8"))
    reencoded = json.dumps(decoded, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    wall = compute_wall(decoded)
    # The reencoded bytes must equal the encoded bytes for the prefix region.
    assert_prefix_unchanged(encoded, reencoded, wall, prefix_len=len(encoded))
