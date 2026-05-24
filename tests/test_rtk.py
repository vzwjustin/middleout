"""Tests for rtk.py compression."""
from __future__ import annotations

import pytest

from middleout_proxy.rtk import compress_rtk

_LEVELS = ("minimal", "standard", "aggressive")

_PROSE = (
    "The function takes a configuration parameter and returns the implementation result. "
    "Because the documentation is approximately complete, you should run the command "
    "as soon as possible to test the regular expression."
)
_CODE_BLOCK_TEXT = (
    "Outside the fence the function returns config.\n"
    "```\n"
    "def the_function(config):\n"
    "    return function  # untouched\n"
    "```\n"
    "Outside the fence, function becomes fn."
)
_IDENT_TEXT = (
    "Call myFunction and the_function and pkg.mod.foo here. "
    "Visit https://example.com/path for the documentation."
)


@pytest.mark.parametrize("level", _LEVELS)
def test_rtk_deterministic(level: str):
    assert compress_rtk(_PROSE, level) == compress_rtk(_PROSE, level)


@pytest.mark.parametrize("level", _LEVELS)
def test_rtk_preserves_code_fence_contents(level: str):
    out = compress_rtk(_CODE_BLOCK_TEXT, level)
    assert "def the_function(config):" in out
    assert "return function  # untouched" in out


@pytest.mark.parametrize("level", _LEVELS)
def test_rtk_preserves_identifiers(level: str):
    out = compress_rtk(_IDENT_TEXT, level)
    assert "myFunction" in out
    assert "the_function" in out
    assert "pkg.mod.foo" in out


@pytest.mark.parametrize("level", _LEVELS)
def test_rtk_preserves_urls(level: str):
    out = compress_rtk(_IDENT_TEXT, level)
    assert "https://example.com/path" in out


def test_rtk_level_monotonic_savings():
    min_out = compress_rtk(_PROSE, "minimal")
    std_out = compress_rtk(_PROSE, "standard")
    agg_out = compress_rtk(_PROSE, "aggressive")
    assert len(agg_out) <= len(std_out) <= len(min_out)


def test_rtk_minimal_function_and_parameter():
    out = compress_rtk("the function and the parameter", "minimal")
    assert "fn" in out
    assert "param" in out
    assert "function" not in out
    assert "parameter" not in out


def test_rtk_standard_because_and_approximately():
    out = compress_rtk("Run it because approximately every minute.", "standard")
    assert "bc" in out
    assert "~" in out
    assert "because" not in out
    assert "approximately" not in out


def test_rtk_minimal_does_not_apply_standard_abbreviations():
    # "because"/"approximately" only appear at standard+.
    out = compress_rtk("Run it because approximately every minute.", "minimal")
    assert "because" in out
    assert "approximately" in out


def test_rtk_aggressive_asap_and_regex():
    out = compress_rtk(
        "Please do this as soon as possible using a regular expression engine.",
        "aggressive",
    )
    assert "ASAP" in out
    assert "regex" in out
    # The original long phrases should be gone.
    assert "as soon as possible" not in out
    assert "regular expression" not in out


def test_rtk_standard_does_not_apply_aggressive_phrases():
    out = compress_rtk(
        "Please do this as soon as possible using a regular expression.",
        "standard",
    )
    # Aggressive-only phrases are intact at standard level.
    assert "as soon as possible" in out
    assert "regular expression" in out


def test_rtk_invalid_level_raises():
    with pytest.raises(ValueError, match="rtk level"):
        compress_rtk("hello", "garbage")


@pytest.mark.parametrize("level", _LEVELS)
def test_rtk_empty_input_returns_empty(level: str):
    assert compress_rtk("", level) == ""
