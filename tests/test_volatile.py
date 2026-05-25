"""Volatile-tail compressor tests.

Locks down the contract between cache_wall and the compressor:
- protected blocks (left of the wall) never reach the compressor
- volatile text blocks DO reach the compressor
- different block shapes (string content, text block, tool_result string,
  tool_result list) all get compressed correctly
- the original payload is left untouched (deepcopy semantics)
- no-win compressions are recorded but don't mutate the payload
"""

from __future__ import annotations

import copy
from dataclasses import dataclass


from middleout_proxy.cache_wall import compute_wall
from middleout_proxy.lingua import LinguaResult
from middleout_proxy.volatile import compress_volatile_tail


@dataclass
class _StubLingua:
    """Drop-in stand-in for LinguaCompressor that the tests can program."""

    transform: callable

    def compress(self, text: str, *, ratio=None) -> LinguaResult:
        out = self.transform(text)
        return LinguaResult(
            text=out,
            chars_in=len(text),
            chars_out=len(out),
            _original=text,
        )


def _shrink_to_half(text: str) -> str:
    if len(text) < 4:
        return text
    return text[: len(text) // 2]


def _ephemeral() -> dict:
    return {"type": "ephemeral"}


def _t(s: str, **extra) -> dict:
    return {"type": "text", "text": s, **extra}


# -- protection ----------------------------------------------------------------


def test_protected_blocks_are_never_compressed() -> None:
    payload = {
        "system": [_t("system prompt with " + "x" * 200, cache_control=_ephemeral())],
        "messages": [
            {"role": "user", "content": [_t("first user " + "y" * 200)]},
            {"role": "assistant", "content": [_t("reply " + "z" * 200)]},
        ],
    }
    original = copy.deepcopy(payload)
    wall = compute_wall(payload)
    lingua = _StubLingua(transform=_shrink_to_half)

    new_payload, audit = compress_volatile_tail(payload, wall=wall, lingua=lingua)

    # Original payload untouched (deepcopy default).
    assert payload == original

    # System block is identical in the new payload.
    assert new_payload["system"] == original["system"]

    # Message text blocks ARE shrunk.
    assert len(new_payload["messages"][0]["content"][0]["text"]) < len(
        original["messages"][0]["content"][0]["text"]
    )
    assert len(new_payload["messages"][1]["content"][0]["text"]) < len(
        original["messages"][1]["content"][0]["text"]
    )

    assert audit.touched
    assert audit.chars_saved > 0


def test_wall_in_messages_protects_earlier_message() -> None:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [_t("early " + "a" * 200, cache_control=_ephemeral())],
            },
            {"role": "assistant", "content": [_t("late " + "b" * 200)]},
        ],
    }
    original = copy.deepcopy(payload)
    wall = compute_wall(payload)
    lingua = _StubLingua(transform=_shrink_to_half)

    new_payload, audit = compress_volatile_tail(payload, wall=wall, lingua=lingua)

    # First message protected.
    assert new_payload["messages"][0] == original["messages"][0]
    # Second message compressed.
    assert len(new_payload["messages"][1]["content"][0]["text"]) < len(
        original["messages"][1]["content"][0]["text"]
    )
    # Audit only records the volatile (second) block.
    assert all(e.msg_idx == 1 for e in audit.events)


# -- block shapes --------------------------------------------------------------


def test_string_message_content_compressed() -> None:
    payload = {
        "messages": [
            {"role": "user", "content": "user " + "x" * 200},
        ],
    }
    wall = compute_wall(payload)
    lingua = _StubLingua(transform=_shrink_to_half)

    new_payload, audit = compress_volatile_tail(payload, wall=wall, lingua=lingua)

    assert isinstance(new_payload["messages"][0]["content"], str)
    assert len(new_payload["messages"][0]["content"]) < len(payload["messages"][0]["content"])
    assert audit.touched


def test_tool_result_string_content_compressed() -> None:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "result " + "x" * 200},
                ],
            },
        ],
    }
    wall = compute_wall(payload)
    lingua = _StubLingua(transform=_shrink_to_half)

    new_payload, audit = compress_volatile_tail(payload, wall=wall, lingua=lingua)

    tr = new_payload["messages"][0]["content"][0]
    assert tr["type"] == "tool_result"
    assert tr["tool_use_id"] == "t1"
    assert isinstance(tr["content"], str)
    assert len(tr["content"]) < len(payload["messages"][0]["content"][0]["content"])
    assert audit.touched


def test_tool_result_list_content_compresses_largest_text() -> None:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {"type": "text", "text": "tiny"},
                            {"type": "text", "text": "huge " + "x" * 300},
                            {"type": "text", "text": "medium " + "y" * 50},
                        ],
                    },
                ],
            },
        ],
    }
    wall = compute_wall(payload)
    lingua = _StubLingua(transform=_shrink_to_half)

    new_payload, audit = compress_volatile_tail(payload, wall=wall, lingua=lingua)

    subs = new_payload["messages"][0]["content"][0]["content"]
    # Only the largest sub-block (index 1) compressed.
    assert subs[0]["text"] == "tiny"
    assert subs[2]["text"] == "medium " + "y" * 50
    assert len(subs[1]["text"]) < len("huge " + "x" * 300)
    assert audit.touched


def test_tool_use_blocks_are_not_compressed() -> None:
    payload = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "u1",
                        "name": "search",
                        "input": {"query": "x" * 500},
                    },
                ],
            },
        ],
    }
    original = copy.deepcopy(payload)
    wall = compute_wall(payload)
    lingua = _StubLingua(transform=_shrink_to_half)

    new_payload, audit = compress_volatile_tail(payload, wall=wall, lingua=lingua)
    assert new_payload == original
    assert not audit.touched
    assert audit.blocks_skipped_non_text >= 1


def test_unknown_block_type_is_skipped() -> None:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "url": "https://example/x.png"}],
            },
        ],
    }
    original = copy.deepcopy(payload)
    wall = compute_wall(payload)
    lingua = _StubLingua(transform=_shrink_to_half)

    new_payload, audit = compress_volatile_tail(payload, wall=wall, lingua=lingua)
    assert new_payload == original
    assert not audit.touched


# -- no-op / no-win paths ------------------------------------------------------


def test_no_win_compression_leaves_text_untouched() -> None:
    payload = {
        "messages": [{"role": "user", "content": "hello world"}],
    }
    wall = compute_wall(payload)

    def _identity(text: str) -> str:
        return text

    lingua = _StubLingua(transform=_identity)
    new_payload, audit = compress_volatile_tail(payload, wall=wall, lingua=lingua)
    # No-win is still recorded as an event but the payload is unchanged.
    assert new_payload == payload
    assert not audit.touched
    assert len(audit.events) == 1
    assert audit.events[0].chars_saved == 0


def test_empty_messages_list_is_safe() -> None:
    payload = {"messages": []}
    wall = compute_wall(payload)
    lingua = _StubLingua(transform=_shrink_to_half)
    new_payload, audit = compress_volatile_tail(payload, wall=wall, lingua=lingua)
    assert new_payload == payload
    assert not audit.touched
    assert audit.blocks_inspected == 0


# -- deepcopy semantics --------------------------------------------------------


def test_deepcopy_default_preserves_input_identity() -> None:
    payload = {
        "messages": [{"role": "user", "content": [_t("x" * 500)]}],
    }
    wall = compute_wall(payload)
    lingua = _StubLingua(transform=_shrink_to_half)
    new_payload, _ = compress_volatile_tail(payload, wall=wall, lingua=lingua)
    assert new_payload is not payload
    assert new_payload["messages"][0]["content"] is not payload["messages"][0]["content"]
    # Original text intact.
    assert payload["messages"][0]["content"][0]["text"] == "x" * 500


def test_deepcopy_false_mutates_in_place() -> None:
    payload = {
        "messages": [{"role": "user", "content": [_t("x" * 500)]}],
    }
    wall = compute_wall(payload)
    lingua = _StubLingua(transform=_shrink_to_half)
    new_payload, _ = compress_volatile_tail(
        payload, wall=wall, lingua=lingua, deepcopy_payload=False
    )
    assert new_payload is payload
    assert len(payload["messages"][0]["content"][0]["text"]) < 500


# -- audit ---------------------------------------------------------------------


def test_audit_paths_are_human_readable() -> None:
    payload = {
        "messages": [
            {"role": "user", "content": [_t("x" * 500), _t("y" * 500)]},
            {"role": "assistant", "content": "z" * 500},
        ],
    }
    wall = compute_wall(payload)
    lingua = _StubLingua(transform=_shrink_to_half)
    _, audit = compress_volatile_tail(payload, wall=wall, lingua=lingua)
    paths = sorted(e.path for e in audit.events)
    assert paths == [
        "messages[0].content[0]",
        "messages[0].content[1]",
        "messages[1].content[0]",
    ]


def test_audit_chars_saved_sums_per_block() -> None:
    payload = {
        "messages": [
            {"role": "user", "content": [_t("a" * 400), _t("b" * 200)]},
        ],
    }
    wall = compute_wall(payload)
    lingua = _StubLingua(transform=_shrink_to_half)
    _, audit = compress_volatile_tail(payload, wall=wall, lingua=lingua)
    # 400 -> 200 (saved 200) + 200 -> 100 (saved 100) = 300
    assert audit.chars_saved == 300


# -- prefix-byte invariant (end-to-end) ---------------------------------------


def test_compression_preserves_prefix_bytes_across_full_payload() -> None:
    """The complete invariant: serializing the input and the output to
    canonical JSON yields identical bytes for the cache-stable prefix
    region. This is what actually matters to Anthropic's prompt cache."""
    import json

    payload = {
        "system": [_t("system " + "s" * 400, cache_control=_ephemeral())],
        "tools": [{"name": "search", "description": "search"}],
        "messages": [
            {"role": "user", "content": [_t("user " + "u" * 400)]},
        ],
    }
    original = copy.deepcopy(payload)
    wall = compute_wall(payload)
    lingua = _StubLingua(transform=_shrink_to_half)
    new_payload, audit = compress_volatile_tail(payload, wall=wall, lingua=lingua)
    assert audit.touched

    orig_sys_bytes = json.dumps(
        original["system"], separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    new_sys_bytes = json.dumps(
        new_payload["system"], separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert orig_sys_bytes == new_sys_bytes  # System region byte-identical.

    orig_tools_bytes = json.dumps(
        original["tools"], separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    new_tools_bytes = json.dumps(
        new_payload["tools"], separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert orig_tools_bytes == new_tools_bytes  # Tools region byte-identical.
