from __future__ import annotations

import json

import pytest

from middleout_proxy.json_aware import compress


def test_json_minify_safe_level_roundtrip():
    src = '{"a": 1, "b": [1, 2, 3]}'
    out, stats = compress(src, "safe")
    assert json.loads(out) == json.loads(src)
    assert len(out) < len(src)
    assert stats["blocks_found"] == 1


def test_partial_json_unchanged_at_safe():
    src = '{"a": 1, "b": [1, 2'
    out, stats = compress(src, "safe")
    assert out == src
    assert stats["blocks_found"] == 0


def test_level_monotonicity_in_blocks_found():
    src = (
        "Prose line one.\n\n\n\n\nProse line two.\n\n"
        "```json\n" + '{"a": 1, "b": 2}' + "\n```\nTrailing.\n"
    )
    safe_blocks = compress(src, "safe")[1]["blocks_found"]
    std_blocks = compress(src, "standard")[1]["blocks_found"]
    agg_blocks = compress(src, "aggressive")[1]["blocks_found"]
    assert safe_blocks <= std_blocks <= agg_blocks


def test_idempotence():
    src = '{"a": 1, "b": [1, 2, 3]}\n\n\n\nmore text'
    once, _ = compress(src, "standard")
    twice, _ = compress(once, "standard")
    assert twice == once


def test_fenced_json_minified():
    src = "before\n```json\n" + '{\n  "a": 1,\n  "b": 2\n}' + "\n```\nafter"
    out, stats = compress(src, "safe")
    assert '{"a":1,"b":2}' in out
    assert stats["blocks_found"] == 1


def test_fenced_python_preserves_indentation():
    src = (
        "intro\n"
        "```python\n"
        "def f():\n"
        "    if True:\n"
        "        return 1\n"
        "\n"
        "\n"
        "    return 2\n"
        "```\n"
    )
    out, _ = compress(src, "standard")
    assert "    if True:" in out
    assert "        return 1" in out
    assert "    return 2" in out


def test_aggressive_strips_line_comment():
    src = "```jsonc\n" + '{\n  // a comment\n  "a": 1\n}' + "\n```\n"
    out, stats = compress(src, "aggressive")
    assert '{"a":1}' in out
    assert "// a comment" not in out
    assert stats["blocks_found"] == 1


def test_aggressive_refuses_when_comment_in_string():
    src = '{"url": "https://example.com/path", "n": 1}'
    out, _stats = compress(src, "aggressive")
    assert json.loads(out) == json.loads(src)
    assert "https://example.com/path" in out


def test_unknown_level_raises_value_error():
    with pytest.raises(ValueError):
        compress("anything", "bogus")


def test_empty_string():
    out, stats = compress("", "safe")
    assert out == ""
    assert stats == {"chars_in": 0, "chars_out": 0, "blocks_found": 0}


def test_stats_dict_shape():
    out, stats = compress("hello world", "standard")
    assert set(stats.keys()) == {"chars_in", "chars_out", "blocks_found"}
    assert stats["chars_in"] == len("hello world")
    assert stats["chars_out"] == len(out)


def test_non_json_text_unchanged_at_safe():
    src = "This is just prose with no JSON blocks at all.\nLine two."
    out, stats = compress(src, "safe")
    assert out == src
    assert stats["blocks_found"] == 0
