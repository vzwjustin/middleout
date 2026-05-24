"""Adaptive policy: decide which engines/levels to run based on payload + model.

Pure functions. Stateless. Deterministic. No I/O, no time, no randomness, no env.

Public API:
  should_compress(payload: dict) -> bool
      False when the total payload text is < 2KB.
  decide_levels(payload: dict) -> dict
      Returns a dict of engine settings keyed off context pressure
      (used tokens / model context window).

Context-window table is prefix-based so families like claude-3-5-sonnet match
both `claude-3-5-sonnet-20240620` and `claude-3-5-sonnet-latest`. Unknown
models default to 200k.
"""

from __future__ import annotations

from typing import Any

_MIN_TOTAL_CHARS = 2048

# Approximate chars per token used to estimate context pressure when the caller
# does not provide token counts. Anthropic models avg ~4 chars per BPE token in
# English; we keep this conservative.
_CHARS_PER_TOKEN = 4

# Prefix -> context window (in tokens). PREFIX matching, ordered longest first.
_MODEL_CONTEXT: tuple[tuple[str, int], ...] = (
    ("claude-3-5-sonnet", 200_000),
    ("claude-3-7-sonnet", 200_000),
    ("claude-3-7", 200_000),
    ("claude-3-opus", 200_000),
    ("claude-3-haiku", 200_000),
    ("claude-opus-4", 200_000),
    ("claude-sonnet-4", 200_000),
    ("claude-haiku-4", 200_000),
    ("claude-3", 200_000),
)
_DEFAULT_CONTEXT = 200_000


def _model_context(model: str) -> int:
    """Prefix-match the model name to a context window."""
    if not model:
        return _DEFAULT_CONTEXT
    for prefix, ctx in _MODEL_CONTEXT:
        if model.startswith(prefix):
            return ctx
    return _DEFAULT_CONTEXT


def _total_text_chars(payload: dict[str, Any]) -> int:
    """Sum the lengths of every str field that participates in input compression."""
    total = 0
    system = payload.get("system")
    if isinstance(system, str):
        total += len(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                total += len(block["text"])
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if isinstance(block.get("text"), str):
                        total += len(block["text"])
                    inner = block.get("content")
                    if isinstance(inner, str):
                        total += len(inner)
                    elif isinstance(inner, list):
                        for sub in inner:
                            if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                                total += len(sub["text"])
    return total


def should_compress(payload: dict[str, Any]) -> bool:
    """False when total payload text is below the 2KB floor."""
    return _total_text_chars(payload) >= _MIN_TOTAL_CHARS


def _pressure(payload: dict[str, Any]) -> float:
    """Approximate fraction of context window used by the payload text."""
    model = payload.get("model", "") if isinstance(payload, dict) else ""
    ctx = _model_context(model if isinstance(model, str) else "")
    chars = _total_text_chars(payload)
    approx_tokens = chars / _CHARS_PER_TOKEN
    if ctx <= 0:
        return 0.0
    return approx_tokens / ctx


# Pressure tiers, in ascending order of compression aggressiveness.
def _tier(pressure: float) -> str:
    if pressure < 0.40:
        return "lenient"
    if pressure < 0.60:
        return "standard"
    if pressure < 0.80:
        return "aggressive"
    return "max"


_TIER_LEVELS = {
    "lenient": {
        "middle_out": "off",
        "caveman": "lite",
        "rtk": "minimal",
        "json_aware": "safe",
        "lsh": "conservative",
        "jl_dedupe": False,
    },
    "standard": {
        "middle_out": "safe",
        "caveman": "standard",
        "rtk": "standard",
        "json_aware": "standard",
        "lsh": "standard",
        "jl_dedupe": True,
    },
    "aggressive": {
        "middle_out": "safe",
        "caveman": "aggressive",
        "rtk": "aggressive",
        "json_aware": "aggressive",
        "lsh": "aggressive",
        "jl_dedupe": True,
    },
    "max": {
        "middle_out": "aggressive",
        "caveman": "ultra",
        "rtk": "aggressive",
        "json_aware": "aggressive",
        "lsh": "aggressive",
        "jl_dedupe": True,
    },
}


def decide_levels(payload: dict[str, Any]) -> dict[str, Any]:
    """Decide per-engine settings based on payload size and model context pressure."""
    pressure = _pressure(payload)
    tier = _tier(pressure)
    # Return a fresh dict so callers can mutate without affecting the constant.
    return dict(_TIER_LEVELS[tier])
