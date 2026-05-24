from middleout_proxy.engines.comment_strip import compress


_FIXTURE = """Here is some Python code:

```python
#!/usr/bin/env python
# This is a top comment
import os

def foo():
    \"\"\"This is a docstring.

    It spans multiple lines.
    \"\"\"
    x = 1  # trailing comment
    y = "# not a comment in string"
    url = "https://example.com"
    return x + 1

# another comment
```

Now some JS:

```js
// header comment
function bar() {
    var s = "// not a comment";
    return s; // trailing
    /* inline */ var x = 1;
}
/* multi
line
block */
function baz() {}
```

Outside the fence: # do not touch
"""


def test_off_is_identity():
    result = compress(_FIXTURE, level="off")
    assert result.text == _FIXTURE
    assert result.chars_saved == 0


def test_short_input_unchanged():
    short = "no fences here, just prose"
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


def test_does_not_corrupt_unrelated_text():
    """Anything outside a fence — including a leading ``#`` line — is
    preserved character-for-character."""
    result = compress(_FIXTURE, level="aggressive")
    assert "Outside the fence: # do not touch" in result.text
    assert "Here is some Python code:" in result.text
    assert "Now some JS:" in result.text


def test_shebang_preserved():
    result = compress(_FIXTURE, level="aggressive")
    assert "#!/usr/bin/env python" in result.text


def test_comment_marker_inside_string_preserved():
    """Comments inside string literals must NOT be stripped."""
    result = compress(_FIXTURE, level="aggressive")
    assert '"# not a comment in string"' in result.text
    assert '"// not a comment"' in result.text


def test_url_double_slash_preserved():
    """``//`` preceded by ``:`` (URL) must not trigger comment stripping."""
    result = compress(_FIXTURE, level="aggressive")
    assert '"https://example.com"' in result.text


def test_lite_strips_full_line_comments_only():
    result = compress(_FIXTURE, level="lite")
    assert "# This is a top comment" not in result.text
    assert "# another comment" not in result.text
    assert "// header comment" not in result.text
    # Trailing comment must be untouched at lite.
    assert "x = 1  # trailing comment" in result.text
    # Docstring must be intact at lite.
    assert "This is a docstring." in result.text


def test_standard_strips_trailing_comments():
    result = compress(_FIXTURE, level="standard")
    # Trailing # comment gone; bare assignment kept.
    assert "# trailing comment" not in result.text
    assert "x = 1" in result.text
    # Trailing // gone; bare statement kept.
    assert "// trailing" not in result.text
    assert "return s;" in result.text


def test_aggressive_strips_docstrings_and_block_comments():
    result = compress(_FIXTURE, level="aggressive")
    # Python docstring removed.
    assert "This is a docstring." not in result.text
    assert "It spans multiple lines." not in result.text
    # Multi-line C block comment removed.
    assert "multi" not in result.text or "/* multi" not in result.text
    assert "block */" not in result.text


def test_unclosed_fence_passthrough():
    """An unterminated fence must NOT corrupt the document."""
    text = "Before\n```python\n# inside\nstill open\n"
    result = compress(text, level="aggressive")
    # Without a closing fence, we leave the buffered content alone.
    assert "# inside" in result.text
    assert "still open" in result.text
