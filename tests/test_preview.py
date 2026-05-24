from middleout_proxy.config import Settings
from middleout_proxy.preview import preview_compression


_EXPECTED_KEYS = {
    "input_chars",
    "output_chars",
    "chars_saved",
    "pct_saved",
    "events",
    "input_token_estimate",
    "output_token_estimate",
    "protected_blocks",
    "cache_hits",
    "cache_misses",
    "compressed_payload",
}


def test_preview_returns_expected_keys_for_simple_payload():
    settings = Settings(input_compression_enabled=True, jl_dedupe_enabled=False)
    payload = {
        "model": "claude-test",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hello world"}],
    }
    result = preview_compression(payload, settings)
    assert _EXPECTED_KEYS.issubset(result.keys())
    assert isinstance(result["compressed_payload"], dict)
    assert result["events"] == []  # short input, nothing compressed
    assert result["chars_saved"] == 0
    assert 0.0 <= result["pct_saved"] <= 100.0
    assert result["input_token_estimate"] >= 0
    assert result["output_token_estimate"] >= 0


def test_preview_chars_saved_with_tiny_max_text_chars():
    settings = Settings(
        max_text_chars=1024,
        min_omission_chars=200,
        head_fraction=0.5,
        input_compression_enabled=True,
        jl_dedupe_enabled=False,
    )
    payload = {
        "model": "claude-test",
        "messages": [{"role": "user", "content": "x" * 8000}],
    }
    result = preview_compression(payload, settings)
    assert result["chars_saved"] > 0
    assert result["output_chars"] < result["input_chars"]
    assert result["pct_saved"] > 0.0
    assert any(event["mode"] == "middle-out" for event in result["events"])
    compressed_content = result["compressed_payload"]["messages"][0]["content"]
    assert "middle-out compressed locally" in compressed_content


def test_preview_safe_on_empty_payload():
    settings = Settings()
    result = preview_compression({}, settings)
    assert result["input_chars"] >= 0
    assert result["output_chars"] >= 0
    assert result["chars_saved"] == 0
    assert result["pct_saved"] == 0.0
    assert result["events"] == []
    assert result["protected_blocks"] == 0
    assert result["compressed_payload"] == {}


def test_preview_safe_on_non_dict_payload():
    settings = Settings()
    # The spec is "safe on user-supplied payloads"; non-dict input must not blow up.
    result = preview_compression(None, settings)  # type: ignore[arg-type]
    assert result["compressed_payload"] == {}
    assert result["chars_saved"] == 0


def test_preview_uses_token_estimate_module_when_available():
    # The token_estimate module now ships as part of the build; the preview
    # path should use it (no fallback marker) and produce a non-negative
    # estimate for both input and output payloads.
    settings = Settings(input_compression_enabled=False)
    payload = {"model": "claude-test", "messages": [{"role": "user", "content": "hi"}]}
    result = preview_compression(payload, settings)
    assert "token_estimate_method" not in result
    assert result["input_token_estimate"] >= 0
    assert result["output_token_estimate"] >= 0


def test_preview_respects_input_compression_disabled():
    settings = Settings(input_compression_enabled=False)
    payload = {"messages": [{"role": "user", "content": "x" * 9000}]}
    result = preview_compression(payload, settings)
    assert result["chars_saved"] == 0
    assert result["events"] == []
    assert result["compressed_payload"]["messages"][0]["content"] == "x" * 9000
