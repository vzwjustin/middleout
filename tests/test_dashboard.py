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
