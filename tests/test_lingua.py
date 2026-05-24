"""LinguaCompressor wrapper tests.

These tests cover the wrapper logic — pass-through thresholds, error handling,
fail-soft behavior — without actually loading the 200MB LLMLingua-2 model. The
model itself is exercised only by an opt-in integration test (skipped unless
the [lingua] extra is installed and the model is cached locally).
"""

from __future__ import annotations

import sys
import types

import pytest

from middleout_proxy.lingua import (
    LinguaCompressor,
    LinguaNotInstalled,
    LinguaResult,
    LinguaUnavailable,
)


# -- construction / validation -------------------------------------------------


def test_construction_with_default_ratio() -> None:
    c = LinguaCompressor()
    assert c.default_ratio == 0.5
    assert c.is_loaded is False


@pytest.mark.parametrize("bad_ratio", [-0.1, 0.0, 0.04, 0.96, 1.0, 2.0])
def test_construction_rejects_out_of_range_ratio(bad_ratio: float) -> None:
    with pytest.raises(ValueError, match="default_ratio"):
        LinguaCompressor(default_ratio=bad_ratio)


def test_compress_rejects_out_of_range_ratio_per_call() -> None:
    c = LinguaCompressor()
    text = "x" * 1000
    with pytest.raises(ValueError, match="ratio"):
        c.compress(text, ratio=0.99)


# -- pass-through thresholds ---------------------------------------------------


def test_empty_input_returns_empty_with_skip_reason() -> None:
    c = LinguaCompressor()
    r = c.compress("")
    assert r.text == ""
    assert r.chars_in == 0
    assert r.chars_out == 0
    assert r.skipped_reason == "empty"
    assert r.chars_saved == 0


def test_too_small_input_passes_through() -> None:
    c = LinguaCompressor()
    text = "short text under threshold"
    r = c.compress(text)
    assert r.text == text
    assert r.chars_out == len(text)
    assert r.skipped_reason == "too_small"
    assert r.chars_saved == 0
    assert c.is_loaded is False  # Never touched the model.


def test_too_large_input_is_refused() -> None:
    c = LinguaCompressor()
    text = "x" * 300_000
    r = c.compress(text)
    assert r.text == text
    assert r.skipped_reason == "too_large"
    assert r.chars_saved == 0
    assert c.is_loaded is False


# -- import / load error paths -------------------------------------------------


def test_missing_llmlingua_raises_lingua_not_installed(monkeypatch) -> None:
    """When `llmlingua` is not importable, the first non-passthrough compress
    call surfaces a `LinguaNotInstalled` (caught and turned into a no-op
    result with skipped_reason='unavailable')."""
    monkeypatch.setitem(sys.modules, "llmlingua", None)  # forces ImportError
    c = LinguaCompressor()
    text = "x" * 1000  # over the small-input threshold
    r = c.compress(text)
    assert r.text == text
    assert r.skipped_reason == "unavailable"
    assert r.chars_saved == 0

    # The error is also retained for direct introspection (subsequent calls
    # surface the same error fast — no retry storm against HuggingFace).
    with pytest.raises(LinguaNotInstalled):
        c._ensure_model()


def test_model_load_failure_raises_lingua_unavailable(monkeypatch) -> None:
    """If `llmlingua` imports but `PromptCompressor(...)` raises, the wrapper
    surfaces `LinguaUnavailable` and the next compress() call returns a
    skipped result rather than crashing."""
    fake_module = types.ModuleType("llmlingua")

    class _FailingCompressor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("model file corrupted")

    fake_module.PromptCompressor = _FailingCompressor
    monkeypatch.setitem(sys.modules, "llmlingua", fake_module)

    c = LinguaCompressor()
    text = "x" * 1000
    r = c.compress(text)
    assert r.skipped_reason == "unavailable"
    with pytest.raises(LinguaUnavailable):
        c._ensure_model()


# -- successful compression with a stub model ---------------------------------


class _StubPromptCompressor:
    """Stub that mimics LLMLingua-2's compress_prompt signature.

    Returns a dict with `compressed_prompt` (newer LLMLingua-2 versions) or a
    plain string (older versions), depending on `style`. We use this to verify
    the wrapper handles both return shapes.
    """

    def __init__(self, *, style: str = "dict", out_text: str = "compressed") -> None:
        self.style = style
        self.out_text = out_text
        self.calls: list[dict] = []

    def compress_prompt(self, text, *, rate, force_tokens=None):
        self.calls.append({"text": text, "rate": rate, "force_tokens": force_tokens})
        if self.style == "dict":
            return {
                "compressed_prompt": self.out_text,
                "origin_tokens": 100,
                "compressed_tokens": 50,
            }
        if self.style == "str":
            return self.out_text
        if self.style == "no_win":
            # Output >= input length means the wrapper must return the input verbatim.
            return text + "x"
        if self.style == "raises":
            raise RuntimeError("inference failed")
        raise AssertionError(f"unknown style {self.style!r}")


def _install_stub(monkeypatch, stub: _StubPromptCompressor) -> None:
    fake_module = types.ModuleType("llmlingua")

    def _factory(*args, **kwargs):
        return stub

    fake_module.PromptCompressor = _factory
    monkeypatch.setitem(sys.modules, "llmlingua", fake_module)


def test_successful_compression_with_dict_return(monkeypatch) -> None:
    stub = _StubPromptCompressor(style="dict", out_text="short")
    _install_stub(monkeypatch, stub)
    c = LinguaCompressor()
    text = "x" * 1000
    r = c.compress(text, ratio=0.3)
    assert r.text == "short"
    assert r.chars_in == 1000
    assert r.chars_out == 5
    assert r.chars_saved == 995
    assert r.dropped_token_count == 50
    assert r.skipped_reason is None
    # Model loaded once, called once.
    assert c.is_loaded
    assert len(stub.calls) == 1
    assert stub.calls[0]["rate"] == 0.3


def test_successful_compression_with_string_return(monkeypatch) -> None:
    stub = _StubPromptCompressor(style="str", out_text="tiny")
    _install_stub(monkeypatch, stub)
    c = LinguaCompressor()
    text = "y" * 800
    r = c.compress(text)
    assert r.text == "tiny"
    assert r.chars_saved == 796


def test_no_win_returns_input_verbatim(monkeypatch) -> None:
    stub = _StubPromptCompressor(style="no_win")
    _install_stub(monkeypatch, stub)
    c = LinguaCompressor()
    text = "z" * 500
    r = c.compress(text)
    assert r.text == text  # unchanged
    assert r.chars_saved == 0
    assert r.skipped_reason == "no_win"


def test_inference_error_fails_soft(monkeypatch) -> None:
    stub = _StubPromptCompressor(style="raises")
    _install_stub(monkeypatch, stub)
    c = LinguaCompressor()
    text = "q" * 600
    r = c.compress(text)
    assert r.text == text
    assert r.chars_saved == 0
    assert r.skipped_reason == "inference_error"


def test_model_loaded_once_across_calls(monkeypatch) -> None:
    stub = _StubPromptCompressor(style="str", out_text="ok")
    factory_calls = {"count": 0}

    fake_module = types.ModuleType("llmlingua")

    def _factory(*args, **kwargs):
        factory_calls["count"] += 1
        return stub

    fake_module.PromptCompressor = _factory
    monkeypatch.setitem(sys.modules, "llmlingua", fake_module)

    c = LinguaCompressor()
    text = "p" * 600
    c.compress(text)
    c.compress(text)
    c.compress(text)
    assert factory_calls["count"] == 1
    assert len(stub.calls) == 3


def test_reset_drops_loaded_model(monkeypatch) -> None:
    stub = _StubPromptCompressor(style="str", out_text="ok")
    factory_calls = {"count": 0}

    fake_module = types.ModuleType("llmlingua")

    def _factory(*args, **kwargs):
        factory_calls["count"] += 1
        return stub

    fake_module.PromptCompressor = _factory
    monkeypatch.setitem(sys.modules, "llmlingua", fake_module)

    c = LinguaCompressor()
    c.compress("p" * 600)
    assert c.is_loaded is True
    c.reset()
    assert c.is_loaded is False
    c.compress("p" * 600)
    assert factory_calls["count"] == 2  # Re-loaded.


# -- LinguaResult ergonomics ---------------------------------------------------


def test_result_chars_saved_never_negative() -> None:
    r = LinguaResult(text="big", chars_in=3, chars_out=99)
    assert r.chars_saved == 0  # clamped
