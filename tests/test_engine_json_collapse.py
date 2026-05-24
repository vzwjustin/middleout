import json

from middleout_proxy.engines.json_collapse import compress


def _build_fixture() -> str:
    """JSON document exercising every level:
       * 60-item int array — every level above off can collapse arrays of this size
       * 15-item str array  — only aggressive collapses (>=10 threshold)
       * 60-key object      — standard + aggressive (>=50 keys)
       * 30-key object      — only aggressive (>=20 keys)
    """
    data = {
        "big_array": list(range(60)),
        "small_array": [f"s{i}" for i in range(15)],
        "big_obj": {f"k{i}": i for i in range(60)},
        "small_obj": {f"k{i}": i for i in range(30)},
    }
    return json.dumps(data)


_FIXTURE = _build_fixture()


def test_off_is_identity():
    result = compress(_FIXTURE, level="off")
    assert result.text == _FIXTURE
    assert result.chars_saved == 0


def test_short_input_unchanged():
    short = "not json"
    result = compress(short, level="aggressive")
    assert result.text == short
    assert result.chars_saved == 0


def test_deterministic():
    a = compress(_FIXTURE, level="standard")
    b = compress(_FIXTURE, level="standard")
    assert a.text == b.text
    assert a.note == b.note


def test_levels_monotone_or_equal():
    off_r = compress(_FIXTURE, level="off")
    lite_r = compress(_FIXTURE, level="lite")
    std_r = compress(_FIXTURE, level="standard")
    agg_r = compress(_FIXTURE, level="aggressive")
    assert off_r.chars_saved == 0
    assert lite_r.chars_saved >= off_r.chars_saved
    assert std_r.chars_saved >= lite_r.chars_saved
    assert agg_r.chars_saved >= std_r.chars_saved
    assert agg_r.chars_saved > std_r.chars_saved  # nested small object catches


def test_does_not_corrupt_unrelated_text():
    """Non-JSON prose must be returned verbatim."""
    text = "Some prose with [brackets] and {braces} that isn't JSON."
    result = compress(text, level="aggressive")
    assert result.text == text


def test_invalid_json_passes_through():
    """Slightly broken JSON should not raise and should not modify the text."""
    text = '{"a": 1, "b": [1, 2, 3,'  # trailing comma + unterminated
    result = compress(text, level="aggressive")
    assert result.text == text
    assert result.chars_saved == 0


def test_array_collapse_contains_type_counts():
    """Per spec: array marker must include type counts."""
    result = compress(_FIXTURE, level="aggressive")
    assert "items omitted" in result.text
    assert "int=" in result.text
    assert "str=" in result.text


def test_lite_only_catches_large_arrays():
    """Lite must not touch objects, only arrays >= 50 items."""
    result = compress(_FIXTURE, level="lite")
    parsed = json.loads(result.text)
    # The 60-key big_obj is untouched.
    assert len(parsed["big_obj"]) == 60
    # The 30-key small_obj is untouched.
    assert len(parsed["small_obj"]) == 30


def test_standard_collapses_objects_too():
    result = compress(_FIXTURE, level="standard")
    parsed = json.loads(result.text)
    # big_obj of 60 keys collapses: head 5 + omitted marker + tail 3 = 9 keys.
    assert len(parsed["big_obj"]) == 9
    # small_obj of 30 keys stays untouched at standard (threshold 50).
    assert len(parsed["small_obj"]) == 30


def test_aggressive_collapses_more_arrays_and_objects():
    result = compress(_FIXTURE, level="aggressive")
    parsed = json.loads(result.text)
    # small_array of 15 items collapses at aggressive (threshold 10).
    assert len(parsed["small_array"]) == 6  # head 3 + marker + tail 2
    # small_obj of 30 keys collapses at aggressive (threshold 20).
    assert len(parsed["small_obj"]) == 9  # head 5 + marker + tail 3


def test_output_is_valid_json():
    """The collapsed result must itself be parseable JSON so the model can
    interpret it."""
    result = compress(_FIXTURE, level="standard")
    json.loads(result.text)  # must not raise
