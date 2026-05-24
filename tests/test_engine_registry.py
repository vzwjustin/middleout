import pytest

from middleout_proxy.engines import (
    ENGINE_NAMES,
    LEVELS,
    REGISTRY,
    EngineResult,
    apply_engine,
)


_EXPECTED_ENGINES = {
    "stack_trace",
    "log_collapse",
    "diff_compactor",
    "comment_strip",
    "path_collapse",
    "json_collapse",
}


def test_registry_has_all_six_engines():
    assert set(REGISTRY.keys()) == _EXPECTED_ENGINES
    assert set(ENGINE_NAMES) == _EXPECTED_ENGINES
    assert len(REGISTRY) == 6


def test_registry_values_are_callables():
    for name, fn in REGISTRY.items():
        assert callable(fn), f"engine {name} is not callable"


def test_apply_engine_returns_engine_result():
    for name in ENGINE_NAMES:
        result = apply_engine(name, "anything", level="standard")
        assert isinstance(result, EngineResult)


def test_apply_engine_rejects_unknown_engine():
    with pytest.raises(ValueError) as exc_info:
        apply_engine("does_not_exist", "text", level="standard")
    assert "unknown engine" in str(exc_info.value)


def test_apply_engine_rejects_unknown_level():
    for name in ENGINE_NAMES:
        with pytest.raises(ValueError) as exc_info:
            apply_engine(name, "text", level="ultra")
        assert "level" in str(exc_info.value)


def test_apply_engine_off_is_identity_for_every_engine():
    text = (
        "Mixed content: a path /Users/x/y/z, a fenced ```python\n# comment\n```, "
        '{"a":1, "b":[1,2,3]}, some prose and a single line.'
    )
    for name in ENGINE_NAMES:
        result = apply_engine(name, text, level="off")
        assert result.text == text
        assert result.chars_saved == 0
        assert result.original_chars == len(text)
        assert result.compressed_chars == len(text)


def test_levels_constant_is_correct():
    assert LEVELS == ("off", "lite", "standard", "aggressive")


def test_engine_modules_export_name_constant():
    """Each engine module should expose a ``NAME`` constant matching its key
    in REGISTRY — useful for telemetry and debugging."""
    from middleout_proxy.engines import (
        comment_strip,
        diff_compactor,
        json_collapse,
        log_collapse,
        path_collapse,
        stack_trace,
    )

    pairs = [
        (stack_trace, "stack_trace"),
        (log_collapse, "log_collapse"),
        (diff_compactor, "diff_compactor"),
        (comment_strip, "comment_strip"),
        (path_collapse, "path_collapse"),
        (json_collapse, "json_collapse"),
    ]
    for module, expected_name in pairs:
        assert module.NAME == expected_name
        assert REGISTRY[expected_name] is module.compress


def test_engine_result_chars_saved_is_nonnegative():
    """``chars_saved`` clamps to zero even if a (hypothetical) engine makes
    text longer."""
    r = EngineResult(text="x", note="", original_chars=5, compressed_chars=10)
    assert r.chars_saved == 0


def test_engine_result_changed_property():
    same = EngineResult(text="abc", note="", original_chars=3, compressed_chars=3)
    diff = EngineResult(text="ab", note="", original_chars=3, compressed_chars=2)
    assert not same.changed
    assert diff.changed
