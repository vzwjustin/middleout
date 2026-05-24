from middleout_proxy.sim.simhash import (
    hamming_distance,
    simhash64,
    simhash_similarity,
)


def test_empty_text_returns_zero():
    assert simhash64("") == 0
    assert simhash64("   \n   ") == 0


def test_signature_fits_in_64_bits():
    h = simhash64("hello world how are you doing today friend let us check this")
    assert 0 <= h < (1 << 64)


def test_identical_text_gives_identical_hash():
    text = "alpha beta gamma delta epsilon zeta " * 5
    assert simhash64(text) == simhash64(text)


def test_deterministic_across_calls():
    text = "the quick brown fox jumps over the lazy dog "
    runs = [simhash64(text) for _ in range(5)]
    assert len(set(runs)) == 1


def test_near_duplicate_hamming_below_eight():
    # ~1 KB of text + a one-word suffix is a textbook near-duplicate.
    base = " ".join(f"token{i}" for i in range(200))
    near = base + " trailingextra"
    a = simhash64(base)
    b = simhash64(near)
    assert hamming_distance(a, b) < 8


def test_disjoint_text_hamming_above_twentyfour():
    # Two ~1 KB blocks built from disjoint English-ish vocab pools. Sequential
    # numeric token names like "alphaword0…alphaword199" bias SimHash output
    # bits and were avoided here — these word lists land at Hamming > 24
    # reliably in practice.
    a_pool = "cat dog bird fish whale elephant turtle lizard mouse hamster"
    b_pool = "orange apple grape lemon mango banana peach pear cherry plum"
    a_text = (a_pool + " ") * 30
    b_text = (b_pool + " ") * 30
    a = simhash64(a_text)
    b = simhash64(b_text)
    assert hamming_distance(a, b) > 24


def test_hamming_matches_python_popcount():
    # Independent oracle: hamming(a, b) == bin(a ^ b).count('1') restricted to 64 bits.
    a = simhash64("hello world this is the quick brown fox jumping")
    b = simhash64("hello world this is the slow brown dog walking")
    xor = (a ^ b) & ((1 << 64) - 1)
    assert hamming_distance(a, b) == bin(xor).count("1")


def test_similarity_in_unit_interval():
    a = simhash64("foo bar baz qux quux corge grault")
    b = simhash64("foo bar baz qux quux corge garply")
    s = simhash_similarity(a, b)
    assert 0.0 <= s <= 1.0


def test_similarity_complementary_to_hamming():
    # 1 - h/64
    a = simhash64("alpha beta gamma delta epsilon zeta eta")
    b = simhash64("alpha beta gamma delta epsilon zeta theta")
    expected = 1.0 - hamming_distance(a, b) / 64
    assert abs(simhash_similarity(a, b) - expected) < 1e-12


def test_punctuation_ignored_for_simhash():
    # tokenize_words strips punctuation, so adding "." should not change the hash.
    bare = "alpha beta gamma delta epsilon"
    decorated = "alpha, beta. gamma! delta? epsilon."
    assert simhash64(bare) == simhash64(decorated)


def test_hamming_distance_zero_for_equal_hashes():
    h = simhash64("a b c d e f g h i j")
    assert hamming_distance(h, h) == 0
    assert simhash_similarity(h, h) == 1.0


def test_seed_change_produces_different_hash():
    text = "alpha beta gamma delta epsilon zeta eta theta iota"
    a = simhash64(text, seed="middleout-sh-v1")
    b = simhash64(text, seed="something-else")
    assert a != b
