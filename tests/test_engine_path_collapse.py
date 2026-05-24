from middleout_proxy.engines.path_collapse import compress


# 92-char path appearing 6 times — caught at every level
_P_LONG = (
    "/Users/justinadams/Desktop/middleout/middleout-claude-proxy/"
    "src/middleout_proxy/long_path.py"
)
# 64-char path appearing 4 times — caught at standard + aggressive (not lite,
# which needs >=5 occurrences and >=80 chars)
_P_MID = "/Users/justin/Documents/projects/some-medium-name/sub/dir/file.py"
# 41-char path appearing 3 times — caught only at aggressive (standard's
# minimum length is 60 chars). With 3 occurrences, net savings stay positive
# after legend overhead.
_P_SHORT = "/Users/justin/work/short_path/file_xyz.py"


def _build_fixture() -> str:
    chunks: list[str] = []
    for _ in range(6):
        chunks.append("see " + _P_LONG)
    for _ in range(4):
        chunks.append("and " + _P_MID)
    for _ in range(3):
        chunks.append("or " + _P_SHORT)
    chunks.append("Some unrelated text without any path.")
    return "\n".join(chunks)


_FIXTURE = _build_fixture()


def test_off_is_identity():
    result = compress(_FIXTURE, level="off")
    assert result.text == _FIXTURE
    assert result.chars_saved == 0


def test_short_input_unchanged():
    short = "hello world"
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
    # Each step should catch strictly more paths.
    assert std_r.chars_saved > lite_r.chars_saved
    assert agg_r.chars_saved > std_r.chars_saved


def test_does_not_corrupt_unrelated_text():
    """Non-path text remains untouched. The legend gets prepended; the
    trailing prose paragraph survives intact."""
    result = compress(_FIXTURE, level="aggressive")
    assert "Some unrelated text without any path." in result.text


def test_first_occurrence_preserved():
    """Per spec: 'Replace later occurrences with a placeholder alias'.
    The first occurrence of each aliased path must still appear verbatim."""
    result = compress(_FIXTURE, level="aggressive")
    # First occurrence of the long path stays as-is.
    first_pos = result.text.find(_P_LONG)
    assert first_pos != -1
    # And there must be at least one ``<P1>`` later in the text.
    assert "<P1>" in result.text[first_pos + len(_P_LONG) :]


def test_lite_only_catches_very_long_frequent_paths():
    result = compress(_FIXTURE, level="lite")
    # Long path qualifies at lite.
    assert "<P1>" in result.text
    # Mid path does not (only 4 occurrences).
    assert _P_MID in result.text
    # Short path does not (only 41 chars).
    assert _P_SHORT in result.text


def test_legend_listed_in_first_occurrence_order():
    """Aliases must be numbered by first-occurrence position (deterministic)."""
    result = compress(_FIXTURE, level="aggressive")
    # The legend should mention <P1>= before <P2>= before <P3>=.
    p1 = result.text.find("<P1>=")
    p2 = result.text.find("<P2>=")
    p3 = result.text.find("<P3>=")
    assert p1 < p2 < p3
    # And <P1> must map to the long path (first to appear in body).
    legend = result.text.split("\n", 1)[0]
    assert _P_LONG in legend.split("<P2>")[0]


def test_text_without_paths_returns_input():
    text = "Just talking, no slashes. Also CamelCase and dotted.names."
    result = compress(text, level="aggressive")
    assert result.text == text
    assert result.chars_saved == 0
