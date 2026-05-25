"""Cheap, deterministic token estimator.

Better than ``len(text) // 4``: weights letters, digits, punctuation, and
whitespace runs separately, and falls back to ``len / 3.5`` for text that is
mostly uppercase or symbol-heavy (e.g. base64 dumps, code with lots of braces).
Used to forecast LLM costs locally without round-tripping to a real tokenizer.
"""

from __future__ import annotations

from typing import Any

__all__ = ["estimate_tokens", "estimate_tokens_for_payload", "summarize_token_stats"]


def estimate_tokens(text: str) -> int:
    """Return a non-negative token estimate for ``text``.

    Heuristic (all additive, weights chosen empirically against BPE tokenizers):

    * letters / 4 — English-ish base case.
    * digits / 2 — digit-heavy text packs more tokens per character.
    * punctuation chars — every standalone symbol is roughly its own token.
    * whitespace_runs * 0.5 — separators have a small per-token cost.

    Texts that are very uppercase-heavy or symbol-heavy fall back to
    ``len(text) / 3.5`` because BPE-style tokenizers produce many tiny tokens
    for those. ``0`` for empty input, ``>=1`` otherwise.
    """
    if not text:
        return 0

    n = len(text)
    letters = 0
    digits = 0
    punctuation = 0
    upper = 0
    ws_runs = 0

    prev_ws = False
    for ch in text:
        if ch.isspace():
            if not prev_ws:
                ws_runs += 1
            prev_ws = True
            continue
        prev_ws = False
        if ch.isalpha():
            letters += 1
            if ch.isupper():
                upper += 1
        elif ch.isdigit():
            digits += 1
        else:
            punctuation += 1

    upper_heavy = letters > 0 and (upper / letters) > 0.6
    symbol_heavy = punctuation > max(1, int(n * 0.3))
    if upper_heavy or symbol_heavy:
        return max(1, round(n / 3.5))

    base = letters / 4.0 + digits / 2.0 + punctuation + ws_runs * 0.5
    return max(1, round(base))


def estimate_tokens_for_payload(payload: dict[str, Any]) -> int:
    """Sum estimated tokens across the standard Anthropic request shape.

    Walks ``system`` (string or list-of-blocks) plus every ``messages[i].content``
    (string or list-of-blocks). Recurses into ``tool_result`` blocks too. Unknown
    block shapes are ignored rather than rejected.
    """
    if not isinstance(payload, dict):
        return 0
    total = _content_tokens(payload.get("system"))
    messages = payload.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                total += _content_tokens(msg.get("content"))
    return total


def summarize_token_stats(payload: dict[str, Any]) -> dict[str, Any]:
    """Per-block breakdown suitable for logging or dashboards.

    Returned shape::

        {
            "total": int,
            "system": int,
            "messages": [{"role": str, "tokens": int}, ...],
        }
    """
    if not isinstance(payload, dict):
        return {"total": 0, "system": 0, "messages": []}

    system_tokens = _content_tokens(payload.get("system"))
    messages_out: list[dict[str, Any]] = []
    messages = payload.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            messages_out.append(
                {
                    "role": msg.get("role", "unknown"),
                    "tokens": _content_tokens(msg.get("content")),
                }
            )
    total = system_tokens + sum(m["tokens"] for m in messages_out)
    return {
        "total": total,
        "system": system_tokens,
        "messages": messages_out,
    }


def _content_tokens(value: Any) -> int:
    """Recursively estimate tokens for a single content value.

    Handles: ``None``, plain string, list of blocks (each a string or a dict
    with ``text``/``content`` fields, including ``tool_result`` shapes).
    """
    if value is None:
        return 0
    if isinstance(value, str):
        return estimate_tokens(value)
    if isinstance(value, list):
        total = 0
        for block in value:
            if isinstance(block, str):
                total += estimate_tokens(block)
            elif isinstance(block, dict):
                total += _block_tokens(block)
        return total
    return 0


def _block_tokens(block: dict[str, Any]) -> int:
    """Token estimate for one Anthropic content block (text / tool_result / etc)."""
    total = 0
    text = block.get("text")
    if isinstance(text, str):
        total += estimate_tokens(text)
    inner = block.get("content")
    if isinstance(inner, str):
        total += estimate_tokens(inner)
    elif isinstance(inner, list):
        for item in inner:
            if isinstance(item, str):
                total += estimate_tokens(item)
            elif isinstance(item, dict):
                inner_text = item.get("text")
                if isinstance(inner_text, str):
                    total += estimate_tokens(inner_text)
    return total
