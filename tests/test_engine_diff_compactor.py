from middleout_proxy.engines.diff_compactor import compress


def _diff_fixture() -> str:
    """A unified diff with:
       * 25-line unchanged context block (catches at every collapse threshold);
       * one straightforward -/+ change pair;
       * a no-op revert pair (-X then +X) for the aggressive level to drop;
       * a final non-revert -/+ pair.
    """
    ctx = "\n".join(f" context line {i:02d}" for i in range(25))
    return (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,30 +1,32 @@\n"
        + ctx
        + "\n"
        + "-old_line_1\n"
        + "+new_line_1\n"
        + "-no_op_revert\n"
        + "+no_op_revert\n"
        + " mid context\n"
        + "-another_change\n"
        + "+another_change_fixed\n"
    )


_FIXTURE = _diff_fixture()


def test_off_is_identity():
    result = compress(_FIXTURE, level="off")
    assert result.text == _FIXTURE
    assert result.chars_saved == 0


def test_short_input_unchanged():
    short = "x = 1"
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
    assert agg_r.chars_saved > std_r.chars_saved  # revert pair drop


def test_does_not_corrupt_unrelated_text():
    """A text without '@@' must be returned untouched, even if it looks
    diff-ish (e.g. lines starting with space)."""
    text = (
        "Here is some prose talking about diffs.\n"
        " - bullet one\n"
        " - bullet two\n"
        " - bullet three\n"
        "+ added\n"
        "- removed\n"
        "But there is no hunk header so it's not a real diff.\n"
    )
    result = compress(text, level="aggressive")
    assert result.text == text


def test_plus_minus_lines_preserved():
    """Only context (' ') lines may be collapsed; +/- lines must stay."""
    result = compress(_FIXTURE, level="standard")
    assert "-old_line_1" in result.text
    assert "+new_line_1" in result.text
    assert "+another_change_fixed" in result.text
    # Hunk header preserved.
    assert "@@ -1,30 +1,32 @@" in result.text


def test_aggressive_drops_revert_pair():
    result = compress(_FIXTURE, level="aggressive")
    # The (-no_op_revert / +no_op_revert) pair must be gone.
    assert "no_op_revert" not in result.text
    # But the non-revert change must still be present.
    assert "+new_line_1" in result.text
    assert "+another_change_fixed" in result.text


def test_file_headers_not_treated_as_changes():
    """``--- a/foo.py`` and ``+++ b/foo.py`` start with -/+ but they're file
    headers; the revert-pair logic should NOT consume them."""
    result = compress(_FIXTURE, level="aggressive")
    assert "--- a/foo.py" in result.text
    assert "+++ b/foo.py" in result.text


def test_collapse_marker_present_at_standard():
    result = compress(_FIXTURE, level="standard")
    assert "unchanged lines" in result.text
