from __future__ import annotations

import random

import pytest

from middleout_proxy.lsh_dedupe import (
    LSHDedupeIndex,
    _minhash_signature,
    dedupe_blocks,
)


def _make_texts(seed=0, n_words=500, change_ratio=0.02):
    rnd = random.Random(seed)
    words = [f'w{i}' for i in range(n_words)]
    rnd.shuffle(words)
    t1 = ' '.join(words)
    w2 = list(words)
    step = max(1, int(1 / change_ratio))
    for i in range(0, len(w2), step):
        w2[i] = 'Z'
    t2 = ' '.join(w2)
    return t1, t2


def test_identical_text_detected():
    idx = LSHDedupeIndex("conservative")
    t = "the quick brown fox jumps over " * 50
    idx.add(0, t)
    match = idx.find_near_duplicate(t)
    assert match is not None
    assert match[0] == 0
    assert match[1] >= 0.99


def test_high_similarity_caught_by_standard_not_conservative():
    # ~0.93 estimated jaccard: above standard (0.88), below conservative (0.95).
    t1, t2 = _make_texts(seed=1, n_words=500, change_ratio=0.005)
    idx_std = LSHDedupeIndex("standard")
    idx_std.add(0, t1)
    m_std = idx_std.find_near_duplicate(t2)
    assert m_std is not None, "standard should catch ~93-pct similar"
    idx_cons = LSHDedupeIndex("conservative")
    idx_cons.add(0, t1)
    m_cons = idx_cons.find_near_duplicate(t2)
    assert m_cons is None, "conservative should not catch sub-0.95 similarity"


def test_level_monotonicity_in_hit_counts():
    # Hit count should not decrease as level becomes more aggressive.
    t1, t2 = _make_texts(seed=2, n_words=400, change_ratio=0.05)
    blocks = [{"type": "text", "text": t1}, {"type": "text", "text": t2}]
    _, stats_c = dedupe_blocks(list(blocks), level="conservative")
    _, stats_s = dedupe_blocks(list(blocks), level="standard")
    _, stats_a = dedupe_blocks(list(blocks), level="aggressive")
    assert stats_c["replaced"] <= stats_s["replaced"] <= stats_a["replaced"]


def test_protected_indices_never_replaced():
    t = "alpha beta gamma delta epsilon zeta eta theta " * 30
    blocks = [{"type": "text", "text": t} for _ in range(3)]
    new, stats = dedupe_blocks(blocks, level="aggressive", protected={1})
    # Block 1 was protected, so its text must be unchanged.
    assert new[1]["text"] == t
    # Block 0 is the original; block 2 should be replaced.
    assert new[0]["text"] == t
    assert "duplicate of earlier block" in new[2]["text"]
    assert stats["replaced"] == 1
    assert stats["protected"] == 1


def test_minhash_deterministic():
    t = "this is a stable input string that should yield the same signature " * 10
    s1 = _minhash_signature(t)
    s2 = _minhash_signature(t)
    assert s1 == s2


def test_value_error_on_bad_level():
    with pytest.raises(ValueError):
        LSHDedupeIndex("bogus")
    with pytest.raises(ValueError):
        dedupe_blocks([], level="bogus")


def test_empty_list_passes():
    new, stats = dedupe_blocks([], level="standard")
    assert new == []
    assert stats["replaced"] == 0


def test_block_not_deduped_against_itself():
    t = "single occurrence text " * 30
    blocks = [{"type": "text", "text": t}]
    new, stats = dedupe_blocks(blocks, level="aggressive")
    assert new[0]["text"] == t
    assert stats["replaced"] == 0


def test_marker_contains_chars_and_similarity():
    t = "repeated content for dedupe marker check " * 30
    blocks = [{"type": "text", "text": t}, {"type": "text", "text": t}]
    new, stats = dedupe_blocks(blocks, level="standard")
    marker = new[1]["text"]
    assert "duplicate of earlier block at 0" in marker
    assert "chars" in marker
    assert "similarity" in marker
    assert stats["replaced"] == 1


def test_idempotence_on_second_pass():
    t = "redo me redo me redo me " * 30
    blocks = [{"type": "text", "text": t}, {"type": "text", "text": t}]
    once, _ = dedupe_blocks(blocks, level="standard")
    twice, stats2 = dedupe_blocks(once, level="standard")
    # Second pass should not change anything since the marker is unique short text.
    assert twice == once
    assert stats2["replaced"] == 0


def test_tool_result_blocks_supported():
    t = "tool result text content to dedupe " * 30
    blocks = [
        {"type": "tool_result", "content": t},
        {"type": "tool_result", "content": t},
    ]
    new, stats = dedupe_blocks(blocks, level="standard")
    assert new[0]["content"] == t
    assert "duplicate of earlier block" in new[1]["content"]
    assert stats["replaced"] == 1


def test_threshold_matches_level_config():
    assert LSHDedupeIndex("conservative").threshold == 0.95
    assert LSHDedupeIndex("standard").threshold == 0.88
    assert LSHDedupeIndex("aggressive").threshold == 0.80


