"""Unit tests for the cost module + provider registry scaffolding."""

from __future__ import annotations

import pytest

from middleout_proxy.cost import (
    CostTracker,
    PRICE_TABLE,
    PriceEntry,
    RequestCost,
    estimate,
    extract_usage_from_anthropic,
    lookup_price,
)


# -- PriceEntry -------------------------------------------------------------


def test_price_entry_total_usd_combines_axes() -> None:
    entry = PriceEntry(
        input_per_mtok=3.0,
        output_per_mtok=15.0,
        cache_write_per_mtok=3.75,
        cache_read_per_mtok=0.30,
    )
    usd = entry.total_usd(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_write_tokens=10_000,
        cache_read_tokens=50_000,
    )
    # 1M @ $3 + 100k @ $15 + 10k @ $3.75 + 50k @ $0.30
    expected = 3.0 + 1.5 + 0.0375 + 0.015
    assert usd == pytest.approx(expected, rel=1e-6)


def test_price_entry_clamps_negative_tokens() -> None:
    entry = PriceEntry(input_per_mtok=10.0, output_per_mtok=10.0)
    assert entry.total_usd(input_tokens=-5, output_tokens=-10) == 0.0


def test_price_entry_handles_missing_cache_rates() -> None:
    """Providers without cache-tier pricing (e.g. OpenAI) ignore cache token counts."""
    entry = PriceEntry(input_per_mtok=2.50, output_per_mtok=10.0)
    usd = entry.total_usd(
        input_tokens=1_000_000,
        output_tokens=0,
        cache_write_tokens=5_000_000,  # should not contribute
        cache_read_tokens=10_000_000,
    )
    assert usd == pytest.approx(2.50, rel=1e-6)


# -- lookup_price -----------------------------------------------------------


def test_lookup_price_longest_prefix_wins() -> None:
    entry = lookup_price("anthropic", "claude-3-5-sonnet-20240620")
    assert entry is not None
    # Sonnet rates, not Opus.
    assert entry.input_per_mtok == 3.0


def test_lookup_price_unknown_model_returns_none() -> None:
    assert lookup_price("anthropic", "claude-future-mega") is None


def test_lookup_price_unknown_provider_returns_none() -> None:
    assert lookup_price("not-a-provider", "any-model") is None


def test_lookup_price_local_zero_cost() -> None:
    entry = lookup_price("ollama", "llama3.1:70b")
    assert entry is not None
    assert entry.input_per_mtok == 0.0
    assert entry.output_per_mtok == 0.0


def test_lookup_price_custom_table_override() -> None:
    custom_table = {
        ("provider-x", "model-foo"): PriceEntry(input_per_mtok=42.0, output_per_mtok=99.0),
    }
    entry = lookup_price("provider-x", "model-foo-instruct", table=custom_table)
    assert entry is not None
    assert entry.input_per_mtok == 42.0


# -- estimate ---------------------------------------------------------------


def test_estimate_returns_matched_cost_for_known_model() -> None:
    cost = estimate(
        provider="anthropic",
        model="claude-3-5-haiku-20241022",
        input_tokens=10_000,
        output_tokens=2_000,
    )
    assert cost.matched is True
    assert cost.usd > 0
    # 10k @ $0.80 + 2k @ $4.00 = 0.008 + 0.008 = 0.016
    assert cost.usd == pytest.approx(0.016, rel=1e-3)


def test_estimate_unknown_model_returns_zero_unmatched() -> None:
    cost = estimate(
        provider="anthropic",
        model="claude-fictitious",
        input_tokens=1000,
        output_tokens=500,
    )
    assert cost.matched is False
    assert cost.usd == 0.0
    # Token counts are preserved even when no price match exists, so callers
    # can still report usage downstream.
    assert cost.input_tokens == 1000
    assert cost.output_tokens == 500


def test_estimate_returns_request_cost_dataclass() -> None:
    cost = estimate(
        provider="anthropic",
        model="claude-3-haiku-20240307",
        input_tokens=500,
        output_tokens=250,
    )
    d = cost.to_dict()
    assert "usd" in d and "matched" in d
    assert d["provider"] == "anthropic"
    assert d["model"] == "claude-3-haiku-20240307"


# -- extract_usage_from_anthropic ------------------------------------------


def test_extract_usage_from_full_anthropic_response() -> None:
    payload = {
        "id": "msg_1",
        "usage": {
            "input_tokens": 1234,
            "output_tokens": 567,
            "cache_creation_input_tokens": 89,
            "cache_read_input_tokens": 12,
        },
    }
    usage = extract_usage_from_anthropic(payload)
    assert usage == {
        "input_tokens": 1234,
        "output_tokens": 567,
        "cache_write_tokens": 89,
        "cache_read_tokens": 12,
    }


def test_extract_usage_handles_missing_fields() -> None:
    usage = extract_usage_from_anthropic({"usage": {"input_tokens": 100}})
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 0
    assert usage["cache_write_tokens"] == 0
    assert usage["cache_read_tokens"] == 0


def test_extract_usage_handles_no_usage_block() -> None:
    usage = extract_usage_from_anthropic({"id": "msg_1"})
    assert all(v == 0 for v in usage.values())


def test_extract_usage_handles_none_payload() -> None:
    usage = extract_usage_from_anthropic(None)
    assert all(v == 0 for v in usage.values())


def test_extract_usage_handles_garbage_payload() -> None:
    usage = extract_usage_from_anthropic("not-a-dict")  # type: ignore[arg-type]
    assert all(v == 0 for v in usage.values())


def test_extract_usage_clamps_negatives_and_strings() -> None:
    payload = {
        "usage": {
            "input_tokens": -5,
            "output_tokens": "abc",  # malformed
            "cache_creation_input_tokens": None,
        },
    }
    usage = extract_usage_from_anthropic(payload)
    assert usage["input_tokens"] == 0  # clamped
    assert usage["output_tokens"] == 0  # malformed -> 0
    assert usage["cache_write_tokens"] == 0  # None -> 0


# -- CostTracker -----------------------------------------------------------


def test_cost_tracker_records_and_snapshots() -> None:
    tracker = CostTracker()
    tracker.record(RequestCost(
        provider="anthropic", model="claude-3-5-sonnet",
        input_tokens=1000, output_tokens=500, usd=0.012, matched=True,
    ))
    tracker.record(RequestCost(
        provider="anthropic", model="claude-3-5-sonnet",
        input_tokens=2000, output_tokens=1000, usd=0.024, matched=True,
    ))
    snap = tracker.snapshot()
    assert snap["total_requests"] == 2
    assert snap["total_usd"] == pytest.approx(0.036, rel=1e-6)
    row = snap["by_model"]["anthropic:claude-3-5-sonnet"]
    assert row["requests"] == 2
    assert row["input_tokens"] == 3000
    assert row["output_tokens"] == 1500


def test_cost_tracker_separates_by_model() -> None:
    tracker = CostTracker()
    tracker.record(RequestCost(provider="anthropic", model="m1", usd=1.0, matched=True))
    tracker.record(RequestCost(provider="anthropic", model="m2", usd=2.0, matched=True))
    snap = tracker.snapshot()
    assert snap["by_model"]["anthropic:m1"]["usd"] == 1.0
    assert snap["by_model"]["anthropic:m2"]["usd"] == 2.0


def test_cost_tracker_counts_unmatched() -> None:
    tracker = CostTracker()
    tracker.record(RequestCost(provider="anthropic", model="m", usd=0.0, matched=False))
    tracker.record(RequestCost(provider="anthropic", model="m", usd=0.0, matched=False))
    tracker.record(RequestCost(provider="anthropic", model="m", usd=1.0, matched=True))
    snap = tracker.snapshot()
    assert snap["unmatched_requests"] == 2
    assert snap["total_requests"] == 3


def test_cost_tracker_reset_clears_everything() -> None:
    tracker = CostTracker()
    tracker.record(RequestCost(provider="x", model="y", usd=5.0, matched=True))
    tracker.reset()
    snap = tracker.snapshot()
    assert snap["total_requests"] == 0
    assert snap["total_usd"] == 0.0
    assert snap["by_model"] == {}


# -- Price table sanity ----------------------------------------------------


def test_price_table_has_required_providers() -> None:
    providers = {p for p, _ in PRICE_TABLE.keys()}
    assert "anthropic" in providers
    assert "openai" in providers
    assert "google" in providers


def test_price_table_sonnet_under_opus() -> None:
    """Sanity: Sonnet must be cheaper than Opus on input tokens."""
    sonnet = lookup_price("anthropic", "claude-3-5-sonnet")
    opus = lookup_price("anthropic", "claude-3-opus")
    assert sonnet is not None and opus is not None
    assert sonnet.input_per_mtok < opus.input_per_mtok
    assert sonnet.output_per_mtok < opus.output_per_mtok
