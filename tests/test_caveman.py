"""Tests for caveman.py compression."""
from __future__ import annotations

import pytest

from middleout_proxy.caveman import compress_caveman

_LEVELS = ("lite", "standard", "aggressive", "ultra")

_PROSE = (
    "Hello, could you please make sure to call the function with the parameter? "
    "Actually, it really is very important that you should return the value of "
    "the implementation. Thanks!"
)
_CODE_BLOCK_TEXT = (
    "Outside the fence we drop articles.\n"
    "```\n"
    "the function inside the fence is preserved verbatim\n"
    "```\n"
    "Outside the fence again, the article goes away."
)
_IDENT_TEXT = (
    "Call myFunction and the_function and pkg.mod.foo here. "
    "Visit https://example.com/path for the documentation."
)


@pytest.mark.parametrize("level", _LEVELS)
def test_caveman_deterministic(level: str):
    assert compress_caveman(_PROSE, level) == compress_caveman(_PROSE, level)


@pytest.mark.parametrize("level", _LEVELS)
def test_caveman_no_longer_than_input(level: str):
    out = compress_caveman(_PROSE, level)
    assert len(out) <= len(_PROSE)


@pytest.mark.parametrize("level", _LEVELS)
def test_caveman_preserves_code_fence_contents(level: str):
    out = compress_caveman(_CODE_BLOCK_TEXT, level)
    # Content inside ``` ... ``` must be untouched.
    assert "the function inside the fence is preserved verbatim" in out
    # Outside the fence "the" should be dropped at every level (article in lowercase).
    # We don't assert here on outside text; the dedicated level tests below cover it.


@pytest.mark.parametrize("level", _LEVELS)
def test_caveman_preserves_urls(level: str):
    out = compress_caveman(_IDENT_TEXT, level)
    assert "https://example.com/path" in out


@pytest.mark.parametrize("level", _LEVELS)
def test_caveman_preserves_identifiers(level: str):
    out = compress_caveman(_IDENT_TEXT, level)
    assert "myFunction" in out
    assert "the_function" in out
    assert "pkg.mod.foo" in out


def test_caveman_invalid_level_raises():
    with pytest.raises(ValueError, match="caveman level"):
        compress_caveman("anything goes", "bogus")


@pytest.mark.parametrize("level", _LEVELS)
def test_caveman_empty_input_returns_empty(level: str):
    assert compress_caveman("", level) == ""


def test_caveman_lite_drops_the_and_a_articles():
    out = compress_caveman("I see the cat and a dog walking down the road.", "lite")
    # Lowercase articles "the" and "a" should be dropped.
    assert " the " not in f" {out} "
    assert " a " not in f" {out} "
    # Content words must still be present.
    assert "cat" in out
    assert "dog" in out


def test_caveman_standard_drops_please_pleasantry():
    out = compress_caveman("Please tell me the answer right away.", "standard")
    # Standard level removes "please" via the pleasantry regex (case-insensitive).
    assert "please" not in out.lower()
    assert "answer" in out


def test_caveman_lite_keeps_please():
    # Lite level does NOT touch pleasantries; only filler + articles.
    out = compress_caveman("Please tell me the answer.", "lite")
    assert "Please" in out


def test_caveman_aggressive_abbreviates_function_to_fn():
    out = compress_caveman("Then call the function and inspect the result.", "aggressive")
    # "function" abbreviated to "fn".
    assert " fn" in out or out.startswith("fn") or " fn " in f" {out} "
    assert "function" not in out


def test_caveman_aggressive_below_does_not_abbreviate():
    # standard does NOT touch "function".
    out = compress_caveman("Then call the function and inspect the result.", "standard")
    assert "function" in out


def test_caveman_ultra_drops_is_and_are():
    out = compress_caveman("This is fine and those are great results.", "ultra")
    # "is" and "are" should both be dropped at ultra (when lowercase).
    assert " is " not in f" {out} "
    assert " are " not in f" {out} "
    assert "fine" in out
    assert "great" in out


def test_caveman_aggressive_does_not_drop_is_and_are():
    out = compress_caveman("This is fine and those are great.", "aggressive")
    # Below ultra, copulas are not dropped.
    assert " is " in f" {out} "
    assert " are " in f" {out} "


def test_caveman_levels_monotonic_savings():
    # Saving must be monotonically non-decreasing along the level chain.
    sizes = {lvl: len(compress_caveman(_PROSE, lvl)) for lvl in _LEVELS}
    assert sizes["lite"] >= sizes["standard"] >= sizes["aggressive"] >= sizes["ultra"]
