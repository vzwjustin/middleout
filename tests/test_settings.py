"""Tests for config.py: _bool_env / _int_env / _float_env parsing + load_settings validation."""
from __future__ import annotations

import importlib
from typing import Callable

import pytest

from middleout_proxy import config as config_module
from middleout_proxy.config import Settings, _bool_env, _float_env, _int_env, load_settings


@pytest.fixture
def reload_config(monkeypatch: pytest.MonkeyPatch) -> Callable[..., object]:
    """Reload the config module with the given env overrides applied.

    The module is reloaded once more at teardown so subsequent tests see the
    original env-driven defaults.
    """

    def _reload(**env_overrides: str) -> object:
        for name, value in env_overrides.items():
            monkeypatch.setenv(name, value)
        return importlib.reload(config_module)

    yield _reload
    # monkeypatch undoes env overrides automatically; reload to refresh class defaults.
    importlib.reload(config_module)


# ---------------------------------------------------------------------------
# _bool_env
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["true", "1", "yes", "on", "TRUE", "YES", "True"])
def test_bool_env_true_values(monkeypatch: pytest.MonkeyPatch, raw: str):
    monkeypatch.setenv("X_TEST_BOOL", raw)
    assert _bool_env("X_TEST_BOOL", default=False) is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "FALSE", ""])
def test_bool_env_false_values(monkeypatch: pytest.MonkeyPatch, raw: str):
    monkeypatch.setenv("X_TEST_BOOL", raw)
    assert _bool_env("X_TEST_BOOL", default=True) is False


def test_bool_env_missing_returns_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("X_TEST_BOOL", raising=False)
    assert _bool_env("X_TEST_BOOL", default=True) is True
    assert _bool_env("X_TEST_BOOL", default=False) is False


# ---------------------------------------------------------------------------
# _int_env
# ---------------------------------------------------------------------------

def test_int_env_parses_integer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("X_TEST_INT", "42")
    assert _int_env("X_TEST_INT", default=0) == 42


def test_int_env_raises_on_garbage(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("X_TEST_INT", "not-an-int")
    with pytest.raises(ValueError, match="X_TEST_INT must be an integer"):
        _int_env("X_TEST_INT", default=0)


def test_int_env_missing_returns_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("X_TEST_INT", raising=False)
    assert _int_env("X_TEST_INT", default=7) == 7


def test_int_env_empty_string_returns_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("X_TEST_INT", "   ")
    assert _int_env("X_TEST_INT", default=11) == 11


# ---------------------------------------------------------------------------
# _float_env
# ---------------------------------------------------------------------------

def test_float_env_parses(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("X_TEST_FLOAT", "3.14")
    assert _float_env("X_TEST_FLOAT", default=0.0) == pytest.approx(3.14)


def test_float_env_raises_on_garbage(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("X_TEST_FLOAT", "nope")
    with pytest.raises(ValueError, match="X_TEST_FLOAT must be a float"):
        _float_env("X_TEST_FLOAT", default=0.0)


# ---------------------------------------------------------------------------
# load_settings
# ---------------------------------------------------------------------------

def test_load_settings_returns_settings_instance():
    settings = load_settings()
    assert isinstance(settings, Settings)


def test_load_settings_raises_when_max_text_chars_below_512(reload_config):
    cfg = reload_config(MIDDLEOUT_MAX_TEXT_CHARS="100")
    with pytest.raises(ValueError, match="MIDDLEOUT_MAX_TEXT_CHARS"):
        cfg.load_settings()


def test_load_settings_raises_when_head_fraction_too_low(reload_config):
    cfg = reload_config(MIDDLEOUT_HEAD_FRACTION="0.01")
    with pytest.raises(ValueError, match="MIDDLEOUT_HEAD_FRACTION"):
        cfg.load_settings()


def test_load_settings_raises_when_head_fraction_too_high(reload_config):
    cfg = reload_config(MIDDLEOUT_HEAD_FRACTION="0.99")
    with pytest.raises(ValueError, match="MIDDLEOUT_HEAD_FRACTION"):
        cfg.load_settings()


def test_load_settings_raises_on_invalid_caveman_level(reload_config):
    cfg = reload_config(MIDDLEOUT_CAVEMAN_LEVEL="bogus")
    with pytest.raises(ValueError, match="MIDDLEOUT_CAVEMAN_LEVEL"):
        cfg.load_settings()


def test_load_settings_raises_on_invalid_rtk_level(reload_config):
    cfg = reload_config(MIDDLEOUT_RTK_LEVEL="bogus")
    with pytest.raises(ValueError, match="MIDDLEOUT_RTK_LEVEL"):
        cfg.load_settings()


def test_load_settings_raises_on_negative_compression_cache_size(reload_config):
    cfg = reload_config(MIDDLEOUT_COMPRESSION_CACHE_SIZE="-1")
    with pytest.raises(ValueError, match="MIDDLEOUT_COMPRESSION_CACHE_SIZE"):
        cfg.load_settings()


def test_load_settings_raises_when_jl_dims_too_low(reload_config):
    cfg = reload_config(MIDDLEOUT_JL_DIMS="8")
    with pytest.raises(ValueError, match="MIDDLEOUT_JL_DIMS"):
        cfg.load_settings()
