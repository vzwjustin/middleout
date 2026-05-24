"""Tests for the inlined dashboard HTML."""
from __future__ import annotations

import html.parser

from middleout_proxy.dashboard import _DASHBOARD_HTML


def test_dashboard_html_is_non_empty_string():
    assert isinstance(_DASHBOARD_HTML, str)
    assert len(_DASHBOARD_HTML) > 0


def test_dashboard_contains_title():
    assert "<title>MiddleOut</title>" in _DASHBOARD_HTML


def test_dashboard_references_stats_endpoint():
    assert "/stats" in _DASHBOARD_HTML


def test_dashboard_references_settings_endpoint():
    assert "/settings" in _DASHBOARD_HTML


def test_dashboard_references_healthz_endpoint():
    assert "/healthz" in _DASHBOARD_HTML


def test_dashboard_html_parses_without_raising():
    parser = html.parser.HTMLParser()
    # html.parser.HTMLParser is intentionally permissive but will raise on
    # invalid constructs depending on the python version's strictness flag.
    parser.feed(_DASHBOARD_HTML)
    parser.close()


def test_dashboard_polls_stats_via_fetch():
    # Look for the script's fetch('/stats') call (string match is sufficient).
    assert "fetch('/stats')" in _DASHBOARD_HTML


def test_dashboard_polls_settings_via_fetch():
    assert "fetch('/settings'" in _DASHBOARD_HTML


def test_dashboard_js_has_no_triple_quote_corruption():
    # Regression: a triple-double-quote sequence inside the rendered JS means a
    # Python triple-quoted-string boundary leaked into the JS payload, which
    # silently breaks the entire dashboard (no toggles bind, no live refresh).
    # JavaScript has no legitimate use for that sequence so any occurrence is
    # a bug.
    bad = '"' * 3
    assert bad not in _DASHBOARD_HTML, (
        "dashboard JS contains a triple-double-quote sequence -- a "
        "Python-string boundary has leaked into the rendered HTML and will "
        "break the entire script"
    )


def test_dashboard_html_escape_helper_is_well_formed():
    # The escapeHtml helper must contain a properly-quoted lookup table.
    # The `"` key must appear as a JS string literal `'"'` (single-quoted),
    # mapping to `&quot;`. The corrupted form had `\"\"` which broke parsing.
    assert "'\"':'&quot;'" in _DASHBOARD_HTML, (
        "escapeHtml lookup table is missing or malformed -- the `\"` -> "
        "`&quot;` mapping must use single-quoted JS string literals"
    )


def test_dashboard_bindings_exist_for_every_toggle():
    # Every visible toggle in the HTML must also have a `bindToggle(...)`
    # call so clicks actually flip state. The DOM id is `t-${key}` and the
    # call is `bindToggle('key'...)`.
    import re
    toggle_ids = set(re.findall(r"id=\"t-([a-z0-9_-]+)\"", _DASHBOARD_HTML))
    bound_keys = set(re.findall(r"bindToggle\('([a-z0-9_-]+)'", _DASHBOARD_HTML))
    missing = toggle_ids - bound_keys
    assert not missing, (
        f"toggles in DOM with no bindToggle() wiring: {sorted(missing)}"
    )


def test_dashboard_readability_v2_palette_contrast():
    # Regression: dashboard text/sub/muted colors must stay at the v2
    # high-contrast values. Reverting these drops legibility on dark bg.
    for needle in (
        "--text:#f5f8fc",
        "--sub:#c2cde0",
        "--muted:#8e98ad",
    ):
        assert needle in _DASHBOARD_HTML, (
            f"dashboard CSS lost the v2 readability palette token: {needle}"
        )


def test_dashboard_has_section_hints():
    # Each major section should carry an .sh-hint subtitle so the user gets
    # a one-line "what is this" without hovering. The CSS rule must also exist.
    assert ".sh .sh-hint" in _DASHBOARD_HTML, "sh-hint CSS rule missing"
    for needle in (
        "Brain <span class=\"sh-hint\"",
        "Cache <span class=\"sh-hint\"",
        "Engines <span class=\"sh-hint\"",
        "Compression <span class=\"sh-hint\"",
        "Traffic <span class=\"sh-hint\"",
    ):
        assert needle in _DASHBOARD_HTML, (
            f"section hint missing in dashboard HTML: {needle}"
        )


def test_dashboard_has_a11y_focus_styles():
    # Keyboard focus visibility for the toggle / level controls. Regression
    # guard so dark-on-dark focus doesn't sneak back.
    assert "*:focus-visible{outline" in _DASHBOARD_HTML, (
        "global :focus-visible rule missing -- keyboard nav loses focus ring"
    )
