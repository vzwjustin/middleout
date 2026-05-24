"""Per-request cost tracker for the brain proxy.

A small pure module that maps `(provider, model, tokens)` to a dollar cost
using a baked-in price table. Pricing is per-million-tokens, split into:

  - ``input_per_mtok`` — uncached input (full prefill price)
  - ``cache_write_per_mtok`` — first-time prompt-cache write surcharge
  - ``cache_read_per_mtok`` — subsequent prompt-cache hit price
  - ``output_per_mtok`` — generated output price

All four are independent so the tracker can charge a request that wrote a new
prefix to the cache differently from one that hit an existing cache row.

The price table is intentionally an in-process constant — operators who need
fresher numbers can monkey-patch :data:`PRICE_TABLE` at startup, or pass an
explicit table to :class:`CostTracker`. No network calls, no remote lookups.

Sources
-------
Numbers reflect publicly listed pricing as of 2025-Q1 / 2025-Q2 for the major
providers we adapt to. They are *guidance*, not contractual — operators billed
on usage receipts should treat this tracker's output as informational.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PriceEntry:
    """One row in the price table. Per-million-token rates in USD."""

    input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float | None = None
    cache_read_per_mtok: float | None = None

    def total_usd(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> float:
        """Estimate dollar cost given a per-axis token breakdown.

        Negative token counts are clamped to zero rather than raising — the
        caller might pass garbage on a malformed upstream response, and we'd
        rather report ``0.00`` than crash the audit log.
        """
        cost = 0.0
        cost += max(0, int(input_tokens)) * self.input_per_mtok / 1_000_000.0
        cost += max(0, int(output_tokens)) * self.output_per_mtok / 1_000_000.0
        if self.cache_write_per_mtok is not None:
            cost += (
                max(0, int(cache_write_tokens))
                * self.cache_write_per_mtok
                / 1_000_000.0
            )
        if self.cache_read_per_mtok is not None:
            cost += (
                max(0, int(cache_read_tokens))
                * self.cache_read_per_mtok
                / 1_000_000.0
            )
        return round(cost, 8)


# Baked-in price table. Keys are `(provider, model_prefix)` — model prefix matches
# the model id with `startswith`, so e.g. `claude-3-5-sonnet-20240620` and
# `claude-3-5-sonnet-latest` both resolve to the same row. Longest prefix wins.
PRICE_TABLE: dict[tuple[str, str], PriceEntry] = {
    # --- Anthropic Claude --------------------------------------------------
    ("anthropic", "claude-opus-4"): PriceEntry(
        input_per_mtok=15.00,
        output_per_mtok=75.00,
        cache_write_per_mtok=18.75,
        cache_read_per_mtok=1.50,
    ),
    ("anthropic", "claude-sonnet-4"): PriceEntry(
        input_per_mtok=3.00,
        output_per_mtok=15.00,
        cache_write_per_mtok=3.75,
        cache_read_per_mtok=0.30,
    ),
    ("anthropic", "claude-haiku-4"): PriceEntry(
        input_per_mtok=0.80,
        output_per_mtok=4.00,
        cache_write_per_mtok=1.00,
        cache_read_per_mtok=0.08,
    ),
    ("anthropic", "claude-3-5-sonnet"): PriceEntry(
        input_per_mtok=3.00,
        output_per_mtok=15.00,
        cache_write_per_mtok=3.75,
        cache_read_per_mtok=0.30,
    ),
    ("anthropic", "claude-3-5-haiku"): PriceEntry(
        input_per_mtok=0.80,
        output_per_mtok=4.00,
        cache_write_per_mtok=1.00,
        cache_read_per_mtok=0.08,
    ),
    ("anthropic", "claude-3-opus"): PriceEntry(
        input_per_mtok=15.00,
        output_per_mtok=75.00,
        cache_write_per_mtok=18.75,
        cache_read_per_mtok=1.50,
    ),
    ("anthropic", "claude-3-haiku"): PriceEntry(
        input_per_mtok=0.25,
        output_per_mtok=1.25,
        cache_write_per_mtok=0.30,
        cache_read_per_mtok=0.03,
    ),
    # --- OpenAI ------------------------------------------------------------
    # GPT-4o family (no prompt cache discounts at time of writing — OpenAI
    # cache pricing isn't surfaced via the chat completions API).
    ("openai", "gpt-4o-mini"): PriceEntry(
        input_per_mtok=0.15, output_per_mtok=0.60,
    ),
    ("openai", "gpt-4o"): PriceEntry(
        input_per_mtok=2.50, output_per_mtok=10.00,
    ),
    ("openai", "gpt-4-turbo"): PriceEntry(
        input_per_mtok=10.00, output_per_mtok=30.00,
    ),
    # --- Google Gemini -----------------------------------------------------
    ("google", "gemini-1.5-flash"): PriceEntry(
        input_per_mtok=0.075, output_per_mtok=0.30,
    ),
    ("google", "gemini-1.5-pro"): PriceEntry(
        input_per_mtok=1.25, output_per_mtok=5.00,
    ),
    ("google", "gemini-2.0-flash"): PriceEntry(
        input_per_mtok=0.10, output_per_mtok=0.40,
    ),
    # --- Local / open-weight (cost is operator hardware, marked 0) ---------
    ("ollama", ""): PriceEntry(input_per_mtok=0.0, output_per_mtok=0.0),
    ("local", ""): PriceEntry(input_per_mtok=0.0, output_per_mtok=0.0),
}


def lookup_price(
    provider: str,
    model: str,
    *,
    table: dict[tuple[str, str], PriceEntry] | None = None,
) -> PriceEntry | None:
    """Return the longest-matching price row, or ``None`` when nothing matches.

    Matching rule: the table key is `(provider, model_prefix)`; we pick the
    entry whose `model_prefix` is the longest prefix of `model` for the given
    `provider`. An empty prefix matches anything (used for "local" models with
    zero cost).
    """
    if table is None:
        table = PRICE_TABLE
    best_prefix = ""
    best_entry: PriceEntry | None = None
    for (prov, prefix), entry in table.items():
        if prov != provider:
            continue
        if not model.startswith(prefix):
            continue
        if best_entry is None or len(prefix) > len(best_prefix):
            best_prefix = prefix
            best_entry = entry
    return best_entry


@dataclass
class RequestCost:
    """One request's cost breakdown. JSON-safe — drop-in for audit logs."""

    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    usd: float = 0.0
    matched: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "usd": round(self.usd, 8),
            "matched": self.matched,
        }


def estimate(
    *,
    provider: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
    table: dict[tuple[str, str], PriceEntry] | None = None,
) -> RequestCost:
    """Compute the dollar cost for one request.

    When no matching row exists, returns a zero-cost record with
    ``matched=False`` — callers can decide whether to flag this or fall back to
    a conservative default. We never raise on unknown models; the operator
    might be exercising a brand-new model id that isn't in the table yet.
    """
    entry = lookup_price(provider, model, table=table)
    if entry is None:
        return RequestCost(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_read_tokens=cache_read_tokens,
            usd=0.0,
            matched=False,
        )
    usd = entry.total_usd(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_read_tokens=cache_read_tokens,
    )
    return RequestCost(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_read_tokens=cache_read_tokens,
        usd=usd,
        matched=True,
    )


def extract_usage_from_anthropic(payload: dict[str, Any] | None) -> dict[str, int]:
    """Parse an Anthropic Messages response and pull token counts out of `usage`.

    Returns a dict with the four token axes our cost model uses. Missing keys
    default to zero so the caller can always pass the result straight into
    :func:`estimate`. Malformed payloads return all zeros — never raises.
    """
    if not isinstance(payload, dict):
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_write_tokens": 0,
            "cache_read_tokens": 0,
        }
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_write_tokens": 0,
            "cache_read_tokens": 0,
        }

    def _int(key: str) -> int:
        try:
            return max(0, int(usage.get(key, 0) or 0))
        except (TypeError, ValueError):
            return 0

    return {
        "input_tokens": _int("input_tokens"),
        "output_tokens": _int("output_tokens"),
        "cache_write_tokens": _int("cache_creation_input_tokens"),
        "cache_read_tokens": _int("cache_read_input_tokens"),
    }


class CostTracker:
    """Process-wide cumulative cost tracker.

    Threadsafe. Records by `(provider, model)` so a multi-provider deployment
    can split out the bill per backend. Never persists to disk — restart wipes
    the counters, matching the audit log's in-memory shape.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_model: dict[tuple[str, str], dict[str, float]] = {}
        self._total_usd: float = 0.0
        self._total_requests: int = 0
        self._unmatched_requests: int = 0

    def record(self, cost: RequestCost) -> None:
        """Add one request's cost to the running totals."""
        key = (cost.provider, cost.model)
        with self._lock:
            row = self._by_model.setdefault(
                key,
                {
                    "requests": 0.0,
                    "input_tokens": 0.0,
                    "output_tokens": 0.0,
                    "cache_write_tokens": 0.0,
                    "cache_read_tokens": 0.0,
                    "usd": 0.0,
                },
            )
            row["requests"] += 1
            row["input_tokens"] += cost.input_tokens
            row["output_tokens"] += cost.output_tokens
            row["cache_write_tokens"] += cost.cache_write_tokens
            row["cache_read_tokens"] += cost.cache_read_tokens
            row["usd"] += cost.usd
            self._total_usd += cost.usd
            self._total_requests += 1
            if not cost.matched:
                self._unmatched_requests += 1

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe summary of cumulative costs."""
        with self._lock:
            return {
                "total_usd": round(self._total_usd, 6),
                "total_requests": self._total_requests,
                "unmatched_requests": self._unmatched_requests,
                "by_model": {
                    f"{p}:{m}": {
                        "requests": int(v["requests"]),
                        "input_tokens": int(v["input_tokens"]),
                        "output_tokens": int(v["output_tokens"]),
                        "cache_write_tokens": int(v["cache_write_tokens"]),
                        "cache_read_tokens": int(v["cache_read_tokens"]),
                        "usd": round(v["usd"], 6),
                    }
                    for (p, m), v in self._by_model.items()
                },
            }

    def reset(self) -> None:
        """Zero all counters. Useful for tests and operator-initiated rollover."""
        with self._lock:
            self._by_model.clear()
            self._total_usd = 0.0
            self._total_requests = 0
            self._unmatched_requests = 0


__all__ = [
    "CostTracker",
    "PRICE_TABLE",
    "PriceEntry",
    "RequestCost",
    "estimate",
    "extract_usage_from_anthropic",
    "lookup_price",
]
