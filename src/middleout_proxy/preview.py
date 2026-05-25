"""Dry-run compression preview.

Pure function. Runs the production :class:`PayloadCompressor` against a payload
and returns a structured summary (sizes, savings, audit events, token
estimates) without touching the network or any global state.

The integration layer is expected to call this from a `/preview` endpoint or
similar inspection tool. The module is intentionally side-effect free so it is
safe to call on user-supplied payloads.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .compression import PayloadCompressor
from .config import Settings


def _serialize_chars(payload: Any) -> int:
    """Length of the JSON serialization. Used as the canonical "size" measure."""
    try:
        return len(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    except (TypeError, ValueError):
        return len(repr(payload))


def _fallback_token_estimate(payload: Any) -> int:
    """Cheap len//4 estimate when the optional :mod:`token_estimate` module is absent."""
    return _serialize_chars(payload) // 4


def preview_compression(
    payload: dict[str, Any],
    settings: Settings,
    *,
    jl_dedupe: bool = True,
    caveman: dict | None = None,
    rtk: dict | None = None,
    input_compression: bool | None = None,
    json_aware: dict | None = None,
    lsh: dict | None = None,
) -> dict[str, Any]:
    """Run compression against ``payload`` and return a structured summary.

    The returned dict is a plain JSON-serializable structure. ``compressed_payload``
    is the post-compression payload (a deep copy — the input is never mutated).

    Args:
        payload: The Anthropic Messages request body to preview. May be empty.
        settings: A :class:`Settings` instance. When ``input_compression`` is
            ``None`` (the default) the preview honors
            ``settings.input_compression_enabled``; pass an explicit bool to
            force the preview to reflect the live runtime toggle instead.
        jl_dedupe: Override for JL-style local dedupe (matches
            :meth:`PayloadCompressor.compress_request_payload`).
        caveman: Optional caveman engine override, same shape as the runtime
            dict in ``server.py`` (``{"enabled": bool, "level": str}``).
        rtk: Optional RTK engine override, same shape.
        input_compression: Optional explicit override of the input-compression
            gate. ``True`` forces compression to run regardless of the static
            setting; ``None`` falls back to ``settings.input_compression_enabled``.
        json_aware: Optional JSON-aware engine override.
        lsh: Optional LSH dedupe engine override.

    Returns:
        A dict with input/output sizes, percentage saved, audit events,
        protected block count, cache hit/miss counters, token estimates, and
        the fully-compressed payload.
    """
    safe_payload: dict[str, Any] = payload if isinstance(payload, dict) else {}

    input_chars = _serialize_chars(safe_payload)

    compressor = PayloadCompressor(settings)
    # Decide the static-gate behavior: when the caller passes an explicit
    # runtime override we honor it; otherwise we fall back to the configured
    # setting so a baseline preview reflects the operator's defaults.
    force = (
        input_compression
        if input_compression is not None
        else settings.input_compression_enabled
    )
    compressed_payload, audit = compressor.compress_request_payload(
        safe_payload,
        endpoint="preview",
        jl_dedupe=jl_dedupe,
        caveman=caveman,
        rtk=rtk,
        json_aware=json_aware,
        lsh=lsh,
        force_enabled=bool(force),
    )

    output_chars = _serialize_chars(compressed_payload)
    chars_saved = max(0, input_chars - output_chars)
    pct_saved = (chars_saved / input_chars * 100.0) if input_chars > 0 else 0.0

    token_method = "default"
    try:
        # Imported lazily so this module works even before token_estimate ships.
        from .token_estimate import estimate_tokens_for_payload  # type: ignore[import-not-found]

        input_token_estimate = int(estimate_tokens_for_payload(safe_payload))
        output_token_estimate = int(estimate_tokens_for_payload(compressed_payload))
    except ImportError:
        input_token_estimate = _fallback_token_estimate(safe_payload)
        output_token_estimate = _fallback_token_estimate(compressed_payload)
        token_method = "fallback"

    result: dict[str, Any] = {
        "input_chars": input_chars,
        "output_chars": output_chars,
        "chars_saved": chars_saved,
        "pct_saved": pct_saved,
        "events": [asdict(event) for event in audit.events],
        "input_token_estimate": input_token_estimate,
        "output_token_estimate": output_token_estimate,
        "protected_blocks": audit.protected_blocks,
        "cache_hits": audit.cache_hits,
        "cache_misses": audit.cache_misses,
        "compressed_payload": compressed_payload,
    }
    if token_method == "fallback":
        result["token_estimate_method"] = "fallback"
    return result


__all__ = ["preview_compression"]
