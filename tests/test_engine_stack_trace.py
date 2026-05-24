from middleout_proxy.engines.stack_trace import compress


_PY_TRACEBACK = """Traceback (most recent call last):
  File "app.py", line 100, in main
    work()
  File "app.py", line 50, in work
    helper()
  File "lib.py", line 10, in helper
    recurse(5)
  File "lib.py", line 5, in recurse
    return recurse(n-1)
  File "lib.py", line 5, in recurse
    return recurse(n-1)
  File "lib.py", line 5, in recurse
    return recurse(n-1)
  File "lib.py", line 5, in recurse
    return recurse(n-1)
  File "lib.py", line 5, in recurse
    return recurse(n-1)
  File "lib.py", line 7, in recurse
    boom()
  File "lib.py", line 22, in boom
    raise RuntimeError("boom")
RecursionError: maximum recursion depth exceeded"""


_JAVA_TRACEBACK = """Exception in thread "main" java.lang.NullPointerException
\tat com.example.Inner.step(Inner.java:42)
\tat com.example.Inner.step(Inner.java:42)
\tat com.example.Inner.step(Inner.java:42)
\tat com.example.Inner.step(Inner.java:42)
\tat com.example.Outer.run(Outer.java:10)
\tat com.example.Main.main(Main.java:6)"""


def test_off_is_identity():
    result = compress(_PY_TRACEBACK, level="off")
    assert result.text == _PY_TRACEBACK
    assert result.chars_saved == 0
    assert result.original_chars == len(_PY_TRACEBACK)
    assert result.compressed_chars == len(_PY_TRACEBACK)


def test_short_input_unchanged():
    short = "just a regular sentence with no traceback"
    result = compress(short, level="aggressive")
    assert result.text == short
    assert result.chars_saved == 0


def test_deterministic():
    r1 = compress(_PY_TRACEBACK, level="standard")
    r2 = compress(_PY_TRACEBACK, level="standard")
    assert r1.text == r2.text
    assert r1.note == r2.note


def test_levels_monotone_or_equal():
    off_r = compress(_PY_TRACEBACK, level="off")
    lite_r = compress(_PY_TRACEBACK, level="lite")
    std_r = compress(_PY_TRACEBACK, level="standard")
    agg_r = compress(_PY_TRACEBACK, level="aggressive")
    assert off_r.chars_saved == 0
    assert lite_r.chars_saved >= off_r.chars_saved
    assert std_r.chars_saved >= lite_r.chars_saved
    assert agg_r.chars_saved >= std_r.chars_saved
    # The aggressive pass should produce strictly more savings than off.
    assert agg_r.chars_saved > 0


def test_does_not_corrupt_unrelated_text():
    surround = "Some prose before.\n\n"
    payload = surround + _PY_TRACEBACK + "\n\nSome prose after."
    result = compress(payload, level="aggressive")
    assert result.text.startswith(surround)
    assert result.text.endswith("Some prose after.")


def test_collapses_python_recursion_at_standard():
    result = compress(_PY_TRACEBACK, level="standard")
    assert "[... 5 identical frames collapsed ...]" in result.text
    assert "collapsed" in result.note
    # Frames outside the recursion run are preserved verbatim.
    assert 'File "app.py", line 100, in main' in result.text
    assert 'File "lib.py", line 22, in boom' in result.text


def test_collapses_java_trace():
    result = compress(_JAVA_TRACEBACK, level="standard")
    assert "[... 4 identical frames collapsed ...]" in result.text
    assert "com.example.Outer.run" in result.text
    assert "com.example.Main.main" in result.text


def test_aggressive_truncates_long_block():
    """At aggressive level, contiguous frame blocks > 5 units get truncated."""
    result = compress(_PY_TRACEBACK, level="aggressive")
    # Truncation marker should be present.
    assert "truncated trace" in result.text
    # Hidden middle frames should no longer appear.
    assert 'File "lib.py", line 10, in helper' not in result.text


def test_no_traceback_returns_input_unchanged():
    text = (
        "I'm just talking about a function called recurse, but this is not a "
        "stack trace. It mentions 'File some.py' but it isn't in traceback "
        "shape because the line doesn't match the regex."
    )
    result = compress(text, level="aggressive")
    assert result.text == text
    assert result.chars_saved == 0


def test_invalid_level_raises():
    import pytest

    with pytest.raises(ValueError):
        compress("anything", level="ultra")
