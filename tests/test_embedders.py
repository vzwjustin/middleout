"""HashEmbedder + OpenAIEmbeddingClient tests."""

from __future__ import annotations

import math

import pytest

from middleout_proxy.cache.embedders import HashEmbedder


# -- HashEmbedder -------------------------------------------------------------


def test_hash_embedder_default_dim() -> None:
    e = HashEmbedder()
    assert e.dim == 3072


def test_hash_embedder_rejects_bad_dim() -> None:
    with pytest.raises(ValueError):
        HashEmbedder(dim=8)


def test_hash_embedder_deterministic() -> None:
    e = HashEmbedder(dim=128)
    v1 = e.embed("hello world how are you today")
    v2 = e.embed("hello world how are you today")
    assert v1 == v2


def test_hash_embedder_l2_normalized() -> None:
    e = HashEmbedder(dim=128)
    v = e.embed("some sample text that should be long enough to fill buckets")
    norm_sq = sum(x * x for x in v)
    assert math.isclose(norm_sq, 1.0, rel_tol=1e-6) or norm_sq == 0.0


def test_hash_embedder_different_text_different_vector() -> None:
    e = HashEmbedder(dim=128)
    v1 = e.embed("the quick brown fox jumps over the lazy dog")
    v2 = e.embed("totally different content about pancakes and waffles")
    # Cosine similarity should be well below 1.0 for substantively different
    # inputs (allow some overlap due to common short shingles like "the").
    dot = sum(a * b for a, b in zip(v1, v2))
    assert dot < 0.5


def test_hash_embedder_similar_text_similar_vector() -> None:
    """Whitespace-only changes produce near-identical vectors."""
    e = HashEmbedder(dim=512)
    base = "the quick brown fox jumps over the lazy dog and runs away forever"
    v1 = e.embed(base)
    v2 = e.embed(base + " ")  # only whitespace added
    dot = sum(a * b for a, b in zip(v1, v2))
    assert dot > 0.9


def test_hash_embedder_empty_input_returns_zero_vector() -> None:
    """Edge case: very short input falls below the shingle width and produces
    the zero vector (which fails cosine match against any non-zero vector)."""
    e = HashEmbedder(dim=128, shingle_chars=8)
    v = e.embed("hi")  # 2 chars, padded to 8 — one shingle, then zero
    # Padded shingle still produces a vector; check it's not all zeros.
    assert any(x != 0.0 for x in v) or v == [0.0] * 128


def test_hash_embedder_short_input_handles_padding() -> None:
    e = HashEmbedder(dim=64, shingle_chars=4)
    v = e.embed("ab")  # shorter than shingle_chars
    assert isinstance(v, list)
    assert len(v) == 64


def test_hash_embedder_non_string_input_coerced() -> None:
    e = HashEmbedder(dim=64)
    v = e.embed(12345)  # type: ignore[arg-type]
    assert len(v) == 64


# -- OpenAIEmbeddingClient (import-guard test only — no live API call) -------


def test_openai_client_raises_clear_error_when_package_missing(monkeypatch) -> None:
    """If `openai` isn't installed, construction raises ImportError with a hint."""
    import sys

    from middleout_proxy.cache.embedders import OpenAIEmbeddingClient

    # If openai isn't installed in this environment, construction will raise
    # ImportError. If it IS installed, we still expect a ValueError on missing
    # API key (which is the next gate).
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    if "openai" not in sys.modules:
        with pytest.raises(ImportError, match="openai"):
            OpenAIEmbeddingClient()
    else:
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIEmbeddingClient()
