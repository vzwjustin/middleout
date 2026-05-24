from middleout_proxy.jl import RequestSketchIndex
from middleout_proxy.sim.jl_index import HybridSketchIndex


def test_empty_index_returns_none_and_negative_score():
    idx = HybridSketchIndex()
    rec, score = idx.find_best("anything at all")
    assert rec is None
    assert score == -1.0


def test_finds_near_duplicate_record():
    # Distinct corpus + one near-duplicate to retrieve.
    idx = HybridSketchIndex()
    idx.add(
        text=" ".join(f"alpha{i}" for i in range(300)),
        path="A",
        digest="da",
    )
    base_text = " ".join(f"word{i}" for i in range(300))
    idx.add(text=base_text, path="B", digest="db")
    idx.add(
        text=" ".join(f"beta{i}" for i in range(300)),
        path="C",
        digest="dc",
    )

    near = base_text + " trailing addition"
    rec, score = idx.find_best(near)
    assert rec is not None
    assert rec.path == "B"
    assert score > 0.95


def test_falls_back_to_full_scan_for_tiny_corpus():
    # Below small_corpus_cutoff the index does a brute-force JL scan; that
    # should still find a near-duplicate even when MinHash LSH might miss.
    idx = HybridSketchIndex(small_corpus_cutoff=16)
    text_a = " ".join(f"foo{i}" for i in range(200))
    text_b = " ".join(f"bar{i}" for i in range(200))
    idx.add(text=text_a, path="A", digest="da")
    idx.add(text=text_b, path="B", digest="db")

    assert len(idx) == 2
    rec, score = idx.find_best(text_a + " a tiny suffix")
    assert rec is not None
    assert rec.path == "A"
    assert score > 0.9


def test_matches_request_sketch_index_score_on_duplicate():
    # On the exact same JL parameters, both indexes should agree on the cosine
    # score of a near-duplicate to within numerical noise.
    base_text = " ".join(f"token{i}" for i in range(400))
    near_text = base_text + " trailing"

    plain = RequestSketchIndex(dims=512, shingle_tokens=5)
    hybrid = HybridSketchIndex(jl_dims=512, jl_shingle_tokens=5)
    # Add several distractors plus the candidate so both indexes go past the
    # small-corpus full-scan threshold.
    for i in range(20):
        text = " ".join(f"distractor{i}_{j}" for j in range(200))
        plain.add(text=text, path=f"d{i}", digest=f"x{i}")
        hybrid.add(text=text, path=f"d{i}", digest=f"x{i}")
    plain.add(text=base_text, path="target", digest="t")
    hybrid.add(text=base_text, path="target", digest="t")

    plain_rec, plain_score = plain.find_best(near_text)
    hybrid_rec, hybrid_score = hybrid.find_best(near_text)
    assert plain_rec is not None
    assert hybrid_rec is not None
    assert plain_rec.path == "target"
    assert hybrid_rec.path == "target"
    assert abs(plain_score - hybrid_score) < 1e-3


def test_len_tracks_added_records():
    idx = HybridSketchIndex()
    assert len(idx) == 0
    idx.add(text="alpha beta gamma delta", path="p", digest="d")
    assert len(idx) == 1
    idx.add(text="kappa lambda mu nu", path="p", digest="d2")
    assert len(idx) == 2


def test_zero_match_returns_negative_for_unrelated_query_in_large_corpus():
    # With a large enough corpus and a truly unrelated query, the LSH may yield
    # no candidates → (None, -1.0). We make 32 docs around a vocab island and
    # query a disjoint vocab to ensure we exceed the small-corpus cutoff.
    idx = HybridSketchIndex(small_corpus_cutoff=16)
    for i in range(32):
        text = " ".join(f"island{j}" for j in range(200))
        idx.add(text=text, path=f"i{i}", digest=str(i))
    rec, score = idx.find_best(" ".join(f"alien{j}" for j in range(200)))
    # We don't strictly require None — LSH might still produce a candidate by
    # accident — but if it does, the cosine score must be very low.
    if rec is None:
        assert score == -1.0
    else:
        assert score < 0.5
