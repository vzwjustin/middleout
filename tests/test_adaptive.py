from __future__ import annotations

import copy

from middleout_proxy.adaptive import decide_levels, should_compress


def _payload(content: str = "", *, model: str = "claude-3-5-sonnet-20240620") -> dict:
    return {"model": model, "messages": [{"role": "user", "content": content}]}


def test_tiny_payload_should_not_compress():
    assert should_compress(_payload("hi")) is False
    assert should_compress({}) is False


def test_should_compress_threshold():
    # 2048 chars is the floor.
    assert should_compress(_payload("x" * 2047)) is False
    assert should_compress(_payload("x" * 2048)) is True


def test_known_model_lenient_at_small_fill():
    levels = decide_levels(_payload("x" * 5000, model="claude-3-5-sonnet-20240620"))
    # 5000 chars ~ 1250 tokens / 200k = 0.6% pressure -> lenient
    assert levels["middle_out"] == "off"
    assert levels["caveman"] == "lite"
    assert levels["rtk"] == "minimal"
    assert levels["json_aware"] == "safe"
    assert levels["lsh"] == "conservative"
    assert levels["jl_dedupe"] is False


def test_same_model_at_90pct_fill_returns_max_levels():
    # 200k tokens * 4 chars/token = 800k chars at 100%. 90% = 720k chars.
    levels = decide_levels(_payload("x" * 720_000, model="claude-3-5-sonnet"))
    assert levels["middle_out"] == "aggressive"
    assert levels["caveman"] == "ultra"
    assert levels["rtk"] == "aggressive"
    assert levels["json_aware"] == "aggressive"
    assert levels["lsh"] == "aggressive"
    assert levels["jl_dedupe"] is True


def test_unknown_model_defaults_to_200k_context():
    # Unknown model + 90% of 200k tokens -> max tier.
    levels = decide_levels(_payload("x" * 720_000, model="some-future-model"))
    assert levels["middle_out"] == "aggressive"


def test_pure_function_no_payload_mutation():
    p = _payload("x" * 10_000, model="claude-opus-4-7-001")
    snapshot = copy.deepcopy(p)
    decide_levels(p)
    should_compress(p)
    assert p == snapshot


def test_decide_levels_keys_present_and_valid():
    levels = decide_levels(_payload("x" * 100_000))
    expected_keys = {"middle_out", "caveman", "rtk", "json_aware", "lsh", "jl_dedupe"}
    assert set(levels.keys()) == expected_keys
    assert levels["middle_out"] in ("off", "safe", "aggressive")
    assert levels["caveman"] in ("lite", "standard", "aggressive", "ultra")
    assert levels["rtk"] in ("minimal", "standard", "aggressive")
    assert levels["json_aware"] in ("safe", "standard", "aggressive")
    assert levels["lsh"] in ("conservative", "standard", "aggressive")
    assert isinstance(levels["jl_dedupe"], bool)


def _rank(levels: dict) -> tuple[int, int, int, int, int]:
    """Map level strings to integers so monotonicity is checkable."""
    middle_out = {"off": 0, "safe": 1, "aggressive": 2}[levels["middle_out"]]
    caveman = {"lite": 0, "standard": 1, "aggressive": 2, "ultra": 3}[levels["caveman"]]
    rtk = {"minimal": 0, "standard": 1, "aggressive": 2}[levels["rtk"]]
    json_aware = {"safe": 0, "standard": 1, "aggressive": 2}[levels["json_aware"]]
    lsh = {"conservative": 0, "standard": 1, "aggressive": 2}[levels["lsh"]]
    return (middle_out, caveman, rtk, json_aware, lsh)


def test_level_monotonicity_vs_pressure():
    # As text grows, no rank should ever decrease.
    sizes = [10_000, 350_000, 500_000, 700_000]
    ranks = [_rank(decide_levels(_payload("x" * n, model="claude-3-5-sonnet"))) for n in sizes]
    for prev, nxt in zip(ranks, ranks[1:]):
        assert all(p <= n for p, n in zip(prev, nxt)), f"non-monotonic: {prev} -> {nxt}"


def test_missing_model_key_handled():
    payload = {"messages": [{"role": "user", "content": "x" * 5000}]}
    levels = decide_levels(payload)
    assert "middle_out" in levels
    # Defaults to 200k context, small fill -> lenient
    assert levels["middle_out"] == "off"


def test_prefix_match_covers_known_families():
    # All these prefixes must yield a known 200k context window so a small fill
    # produces the lenient tier.
    families = [
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
        "claude-3-7-sonnet-2025xx",
        "claude-opus-4-7-001",
        "claude-sonnet-4-5-002",
        "claude-haiku-4-0-001",
    ]
    for model in families:
        levels = decide_levels(_payload("x" * 5000, model=model))
        assert levels["middle_out"] == "off", f"{model} should be lenient on small payload"
