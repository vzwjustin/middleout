from middleout_proxy.engines.log_collapse import compress


def _log_fixture() -> str:
    """A log block with:
       * 12 lines that differ only by an ISO timestamp (standard+ collapses);
       * 6 lines that differ only in numeric tokens (aggressive only);
       * 10 byte-identical lines (every level except off can collapse).
    """
    parts: list[str] = ["Server starting up"]
    parts += [
        f"2023-01-15 10:30:{45 + i:02d} INFO request received from client"
        for i in range(12)
    ]
    parts.append("Heartbeat")
    parts += [f"Processed {n} items in cycle" for n in range(100, 106)]
    parts.append("Heartbeat")
    parts += ["Identical line here"] * 10
    parts.append("Shutdown")
    return "\n".join(parts)


_FIXTURE = _log_fixture()


def test_off_is_identity():
    result = compress(_FIXTURE, level="off")
    assert result.text == _FIXTURE
    assert result.chars_saved == 0


def test_short_input_unchanged():
    short = "single line of output"
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
    assert agg_r.chars_saved > std_r.chars_saved  # numbers should also collapse


def test_does_not_corrupt_unrelated_text():
    """Non-log prose with no repeated lines stays exactly the same."""
    text = (
        "Here is a paragraph that talks about logs but has no repeated lines. "
        "It mentions 2023-01-15 in passing but only once. Done."
    )
    result = compress(text, level="aggressive")
    assert result.text == text


def test_lite_only_collapses_byte_identical_runs():
    """Lite must NOT touch lines that only differ by timestamp."""
    result = compress(_FIXTURE, level="lite")
    # The 10 identical lines should be collapsed.
    assert "Identical line here" in result.text
    assert "[... 8 identical lines collapsed ...]" in result.text
    # But the timestamp-prefixed lines all differ byte-for-byte, so they stay.
    assert "2023-01-15 10:30:45 INFO request received from client" in result.text
    assert "2023-01-15 10:30:56 INFO request received from client" in result.text


def test_standard_collapses_timestamp_runs():
    result = compress(_FIXTURE, level="standard")
    assert "[... 10 identical lines collapsed ...]" in result.text
    # First and last of each run are preserved verbatim.
    assert "2023-01-15 10:30:45 INFO request received from client" in result.text
    assert "2023-01-15 10:30:56 INFO request received from client" in result.text


def test_aggressive_normalizes_numeric_tokens():
    """The 'Processed N items in cycle' run differs only in N."""
    result = compress(_FIXTURE, level="aggressive")
    assert "Processed 100 items in cycle" in result.text
    assert "Processed 105 items in cycle" in result.text
    # Middle entries (101..104) should be collapsed.
    assert "Processed 101 items in cycle" not in result.text
    assert "Processed 104 items in cycle" not in result.text
