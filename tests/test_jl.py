"""Tests for jl.py: tokenize, shingles, signed_jl_projection, cosine, RequestSketchIndex."""
from __future__ import annotations

import math

import pytest

from middleout_proxy.jl import (
    RequestSketchIndex,
    cosine,
    shingles,
    signed_jl_projection,
    tokenize,
)


def test_tokenize_basic():
    toks = tokenize("Hello, World! Foo_bar 42.5")
    assert "hello" in toks
    assert "world" in toks
    assert "foo_bar" in toks  # snake_case identifier kept whole
    assert "42.5" in toks
    # Punctuation tokens are present and lowercased (already lower-only).
    assert "," in toks
    assert "!" in toks
    # All tokens are lower-cased.
    assert all(t == t.lower() for t in toks)


def test_tokenize_empty():
    assert tokenize("") == []


def test_shingles_short_input():
    # Fewer tokens than width: collapses to a single joined shingle.
    out = list(shingles(["a", "b", "c"], 5))
    assert out == ["a b c"]


def test_shingles_exact_width():
    out = list(shingles(["a", "b", "c", "d", "e"], 5))
    assert out == ["a b c d e"]


def test_shingles_count():
    # N=7, K=3 -> N-K+1 = 5 shingles.
    out = list(shingles(["a", "b", "c", "d", "e", "f", "g"], 3))
    assert len(out) == 5
    assert out[0] == "a b c"
    assert out[-1] == "e f g"


def test_shingles_empty_tokens_yields_nothing():
    assert list(shingles([], 5)) == []


def test_signed_jl_projection_deterministic():
    text = "the quick brown fox jumps over the lazy dog"
    a = signed_jl_projection(text, dims=128)
    b = signed_jl_projection(text, dims=128)
    assert a == b


def test_signed_jl_projection_unit_norm():
    text = "hello world this is a non-empty piece of text for sketching"
    v = signed_jl_projection(text, dims=128)
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-9


def test_signed_jl_projection_empty_string():
    v = signed_jl_projection("", dims=64)
    assert len(v) == 64
    assert all(x == 0.0 for x in v)
    # And not unit-normalized (it's zero by definition).
    assert math.sqrt(sum(x * x for x in v)) == 0.0


def test_cosine_identical():
    text = "alpha beta gamma delta epsilon zeta eta theta"
    v = signed_jl_projection(text, dims=64)
    assert abs(cosine(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal_random():
    a = signed_jl_projection(
        "completely different words about boats and oceans and sailing",
        dims=256,
    )
    b = signed_jl_projection(
        "totally unrelated content concerning cooking pasta with fresh tomatoes",
        dims=256,
    )
    assert cosine(a, b) < 0.5


def test_cosine_dim_mismatch_raises():
    with pytest.raises(ValueError):
        cosine((1.0, 0.0), (1.0, 0.0, 0.0))


def test_request_sketch_index_empty_find_best():
    idx = RequestSketchIndex(dims=64, shingle_tokens=5)
    record, score = idx.find_best("anything at all")
    assert record is None
    assert score == -1.0


def test_request_sketch_index_add_then_find():
    idx = RequestSketchIndex(dims=128, shingle_tokens=5)
    text = "this is a moderately long sentence used as a sketch fingerprint test"
    idx.add(text=text, path="messages[0].user.content", digest="deadbeef00000000")
    record, score = idx.find_best(text)
    assert record is not None
    assert record.path == "messages[0].user.content"
    assert record.digest == "deadbeef00000000"
    assert score > 0.99


def test_request_sketch_index_multiple_records_picks_best():
    idx = RequestSketchIndex(dims=256, shingle_tokens=4)
    text_a = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    text_b = "the cat sat on the mat and then it stood up again to look around"
    text_c = (
        "completely different content about marine biology, coral reefs, and tropical fish "
        "swimming in clear warm water near the surface"
    )
    idx.add(text=text_a, path="A", digest="aaaa")
    idx.add(text=text_b, path="B", digest="bbbb")
    idx.add(text=text_c, path="C", digest="cccc")

    # Query identical to A -> should pick A as best.
    record, score = idx.find_best(text_a)
    assert record is not None
    assert record.path == "A"
    assert score > 0.99

    # Query identical to C -> should pick C as best.
    record_c, score_c = idx.find_best(text_c)
    assert record_c is not None
    assert record_c.path == "C"
    assert score_c > 0.99


def test_signed_jl_projection_different_seeds():
    text = "the quick brown fox jumps over the lazy dog"
    a = signed_jl_projection(text, dims=128, seed="seed-one")
    b = signed_jl_projection(text, dims=128, seed="seed-two")
    assert a != b
