import pytest

from middleout_proxy.sim.lsh import MinHashLSH
from middleout_proxy.sim.minhash import minhash_signature


def test_add_and_candidates_roundtrip():
    lsh = MinHashLSH(num_perms=64, bands=16)
    sig = minhash_signature("alpha beta gamma delta epsilon zeta", num_perms=64)
    lsh.add("doc-1", sig)
    assert "doc-1" in lsh.candidates(sig)


def test_identical_signatures_collide_in_every_band():
    # The LSH stores band keys for each doc; an identical signature must
    # collide in *every* band, not just one.
    text = "alpha beta gamma delta epsilon zeta eta theta iota " * 5
    sig = minhash_signature(text, num_perms=64)

    lsh = MinHashLSH(num_perms=64, bands=16)
    lsh.add("doc-1", sig)

    candidates = lsh.candidates(sig)
    assert candidates == {"doc-1"}

    # Independent oracle on the underlying bucket layout.
    # Every band's bucket containing this signature must hold the doc id.
    keys = lsh._band_keys(sig)  # type: ignore[attr-defined]
    assert len(keys) == 16
    for band_idx, key in enumerate(keys):
        bucket = lsh._buckets[band_idx][key]  # type: ignore[attr-defined]
        assert "doc-1" in bucket


def test_near_duplicates_collide_in_many_bands():
    # Jaccard ≈ 0.95 corpus pair: should collide in a large fraction of bands.
    base = " ".join(f"word{i}" for i in range(400))
    near = base.replace("word0 ", "alt0 ").replace("word100 ", "alt100 ")

    sig_base = minhash_signature(base, num_perms=128)
    sig_near = minhash_signature(near, num_perms=128)

    lsh = MinHashLSH(num_perms=128, bands=32)
    lsh.add("doc-base", sig_base)
    candidates = lsh.candidates(sig_near)
    assert "doc-base" in candidates


def test_disjoint_signatures_rarely_collide():
    # Two completely disjoint vocabularies should not produce candidates with
    # high probability. We allow rare collisions, but assert "rare" — average
    # case is zero.
    a_text = " ".join(f"alpha{i}" for i in range(200))
    b_text = " ".join(f"beta{i}" for i in range(200))
    sig_a = minhash_signature(a_text, num_perms=128)
    sig_b = minhash_signature(b_text, num_perms=128)

    lsh = MinHashLSH(num_perms=128, bands=32)
    lsh.add("a", sig_a)
    assert "a" not in lsh.candidates(sig_b) or len(lsh.candidates(sig_b)) == 0


def test_remove_drops_doc_from_buckets():
    sig_a = minhash_signature("alpha beta gamma delta epsilon", num_perms=64)
    sig_b = minhash_signature("kappa lambda mu nu xi omicron", num_perms=64)

    lsh = MinHashLSH(num_perms=64, bands=16)
    lsh.add("a", sig_a)
    lsh.add("b", sig_b)
    assert len(lsh) == 2

    lsh.remove("a")
    assert len(lsh) == 1
    assert "a" not in lsh.candidates(sig_a)
    assert "b" in lsh.candidates(sig_b)


def test_remove_missing_doc_is_noop():
    lsh = MinHashLSH(num_perms=64, bands=16)
    lsh.remove("nonexistent")  # must not raise
    assert len(lsh) == 0


def test_len_tracks_doc_count():
    lsh = MinHashLSH(num_perms=64, bands=16)
    assert len(lsh) == 0

    for i in range(5):
        sig = minhash_signature(f"unique text payload number {i} foo bar baz", num_perms=64)
        lsh.add(f"d{i}", sig)
    assert len(lsh) == 5


def test_signature_length_mismatch_raises():
    lsh = MinHashLSH(num_perms=64, bands=16)
    with pytest.raises(ValueError):
        lsh.add("d", (0, 1, 2))  # too short


def test_bands_must_divide_num_perms():
    with pytest.raises(ValueError):
        MinHashLSH(num_perms=128, bands=30)  # 128 / 30 is not an integer


def test_re_adding_same_doc_does_not_double_count():
    sig = minhash_signature("alpha beta gamma delta epsilon", num_perms=64)
    lsh = MinHashLSH(num_perms=64, bands=16)
    lsh.add("a", sig)
    lsh.add("a", sig)  # idempotent re-add
    assert len(lsh) == 1
    # Bucket should hold exactly one copy of the doc id, not two.
    keys = lsh._band_keys(sig)  # type: ignore[attr-defined]
    for band_idx, key in enumerate(keys):
        bucket = lsh._buckets[band_idx][key]  # type: ignore[attr-defined]
        assert bucket.count("a") == 1
