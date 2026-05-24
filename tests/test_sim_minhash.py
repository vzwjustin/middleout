import pytest

from middleout_proxy.sim.minhash import jaccard_estimate, minhash_signature


_UINT64_MAX = (1 << 64) - 1


def test_empty_text_returns_max_signature():
    sig = minhash_signature("", num_perms=64)
    assert sig == (_UINT64_MAX,) * 64


def test_whitespace_only_text_returns_max_signature():
    # No alphanumeric tokens → no shingles → MAX-valued signature.
    sig = minhash_signature("   \n\t  ", num_perms=32)
    assert sig == (_UINT64_MAX,) * 32


def test_signature_length_matches_num_perms():
    sig = minhash_signature("the quick brown fox jumps over the lazy dog", num_perms=96)
    assert len(sig) == 96
    assert all(isinstance(v, int) for v in sig)
    assert all(0 <= v <= _UINT64_MAX for v in sig)


def test_signature_is_deterministic():
    text = "the quick brown fox jumps over the lazy dog " * 4
    a = minhash_signature(text, num_perms=64)
    b = minhash_signature(text, num_perms=64)
    assert a == b


def test_identical_text_gives_jaccard_1():
    text = "alpha beta gamma delta epsilon zeta " * 30
    a = minhash_signature(text)
    b = minhash_signature(text)
    assert jaccard_estimate(a, b) == 1.0


def test_near_duplicate_jaccard_is_high():
    # 200 unique tokens → 196 unique 5-shingles. Appending " extra suffix"
    # only adds 2 new shingles, so exact Jaccard ≈ 196/198 ≈ 0.99 and MinHash
    # should estimate that within a few percent.
    base = " ".join(f"word{i}" for i in range(200))
    near = base + " extra suffix"
    a = minhash_signature(base, num_perms=128)
    b = minhash_signature(near, num_perms=128)
    assert jaccard_estimate(a, b) >= 0.85


def test_disjoint_text_jaccard_is_low():
    a_text = "alpha beta gamma delta epsilon zeta eta theta " * 30
    b_text = "banana orange guitar river mountain canyon forest desert " * 30
    a = minhash_signature(a_text, num_perms=128)
    b = minhash_signature(b_text, num_perms=128)
    assert jaccard_estimate(a, b) < 0.1


def test_jaccard_estimate_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        jaccard_estimate((1, 2, 3), (1, 2, 3, 4))


def test_num_perms_must_be_positive():
    with pytest.raises(ValueError):
        minhash_signature("hello world", num_perms=0)


def test_punctuation_is_ignored_so_dups_collapse():
    # tokenize_words is alphanumeric-only, so punctuation should not change the sig.
    a = minhash_signature("hello, world! how are you?", num_perms=64)
    b = minhash_signature("hello world how are you", num_perms=64)
    assert jaccard_estimate(a, b) == 1.0


def test_shingle_size_changes_signature():
    text = "alpha beta gamma delta epsilon zeta " * 5
    s3 = minhash_signature(text, num_perms=64, shingle_size=3)
    s5 = minhash_signature(text, num_perms=64, shingle_size=5)
    assert s3 != s5


def test_signature_values_are_64bit_unsigned():
    sig = minhash_signature("hello world goodbye world hi there friend", num_perms=32)
    for v in sig:
        assert 0 <= v < (1 << 64)


def test_seed_change_produces_different_signature():
    text = "one two three four five six seven eight nine ten " * 5
    a = minhash_signature(text, num_perms=64, seed="middleout-mh-v1")
    b = minhash_signature(text, num_perms=64, seed="something-else")
    assert a != b
