"""Prometheus 0.0.4 text-format exporter.

Renders a flat snapshot of audit + compression-cache statistics into the
Prometheus text exposition format. The function is pure — no scraping, no
global registry, no HTTP. The integration layer is expected to call this from
a ``/metrics`` route.

Naming follows Prometheus conventions: counters end with ``_total``, gauges
representing time end with ``_seconds``, and all names use ``snake_case``.
"""

from __future__ import annotations

from typing import Any

from .config import Settings


def _escape_label_value(value: str) -> str:
    """Escape a label value per the Prometheus text format."""
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _get_int(stats: dict[str, Any], key: str, default: int = 0) -> int:
    """Pull an int from ``stats`` tolerating None / missing keys."""
    value = stats.get(key, default)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_float(stats: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = stats.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cache_field(stats: dict[str, Any], key: str, default: int = 0) -> int:
    """Read a cache field that may be top-level (flat merge) or nested under ``result_cache``."""
    if key in stats and stats[key] is not None:
        return _get_int(stats, key, default)
    nested = stats.get("result_cache")
    if isinstance(nested, dict) and key in nested:
        return _get_int(nested, key, default)
    return default


def render_prometheus(stats: dict[str, Any], *, settings: Settings) -> str:
    """Render ``stats`` and ``settings`` to a Prometheus text-format payload.

    Args:
        stats: A merged snapshot of :class:`ProxyStats` and
            :class:`_CompressionResultCache.stats`. Either a flat dict
            (``cache_hits`` etc. at the top level along with ``size``,
            ``max_entries``) or a nested dict where the LRU stats live under
            a ``result_cache`` sub-dict — both shapes are accepted.
        settings: Current :class:`Settings`. Used to expose configured
            toggles (input compression on/off, engine on/off, JL similarity).

    Returns:
        A newline-terminated Prometheus 0.0.4 text exposition body.
    """
    lines: list[str] = []

    def emit_counter(name: str, help_text: str, value: int) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name} {value}")

    def emit_gauge(
        name: str,
        help_text: str,
        value: float | int | str,
        *,
        labels: dict[str, str] | None = None,
        emit_meta: bool = True,
    ) -> None:
        if emit_meta:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
        if labels:
            label_str = ",".join(
                f'{k}="{_escape_label_value(str(v))}"' for k, v in labels.items()
            )
            lines.append(f"{name}{{{label_str}}} {value}")
        else:
            lines.append(f"{name} {value}")

    # --- counters ---------------------------------------------------------
    emit_counter(
        "middleout_requests_total",
        "Total HTTP requests handled by the proxy.",
        _get_int(stats, "requests_total"),
    )
    emit_counter(
        "middleout_compressed_requests_total",
        "Requests that had at least one compression event.",
        _get_int(stats, "compressed_requests"),
    )
    emit_counter(
        "middleout_upstream_errors_total",
        "Requests that failed to reach the upstream cleanly.",
        _get_int(stats, "upstream_errors"),
    )
    emit_counter(
        "middleout_chars_saved_in_total",
        "Characters saved on inbound (request) payloads.",
        _get_int(stats, "chars_saved_in"),
    )
    emit_counter(
        "middleout_chars_saved_out_total",
        "Characters saved on outbound (response) payloads.",
        _get_int(stats, "chars_saved_out"),
    )
    emit_counter(
        "middleout_protected_blocks_total",
        "Blocks skipped to preserve the Anthropic prompt cache.",
        _get_int(stats, "protected_blocks"),
    )
    emit_counter(
        "middleout_cache_hits_total",
        "Local LRU compression cache hits.",
        _get_int(stats, "cache_hits"),
    )
    emit_counter(
        "middleout_cache_misses_total",
        "Local LRU compression cache misses.",
        _get_int(stats, "cache_misses"),
    )

    # --- gauges -----------------------------------------------------------
    emit_gauge(
        "middleout_uptime_seconds",
        "Seconds since the proxy process started.",
        f"{_get_float(stats, 'uptime_s'):.6f}",
    )
    emit_gauge(
        "middleout_cache_size",
        "Current number of entries in the LRU compression cache.",
        _cache_field(stats, "size"),
    )
    emit_gauge(
        "middleout_cache_max_entries",
        "Configured maximum entries in the LRU compression cache.",
        _cache_field(stats, "max_entries"),
    )
    emit_gauge(
        "middleout_input_compression_enabled",
        "Whether the named compression engine is enabled (1 = on, 0 = off).",
        1 if settings.input_compression_enabled else 0,
        labels={"engine": "input"},
    )

    # --- per-engine gauges (one HELP/TYPE block, many labeled values) -----
    lines.append(
        "# HELP middleout_engine_enabled "
        "Whether a named compression engine is enabled (1 = on, 0 = off)."
    )
    lines.append("# TYPE middleout_engine_enabled gauge")
    engine_states: list[tuple[str, bool]] = [
        ("caveman", settings.caveman_enabled),
        ("rtk", settings.rtk_enabled),
        ("jl_dedupe", settings.jl_dedupe_enabled),
        ("output", settings.output_compression_enabled),
    ]
    for engine_name, enabled in engine_states:
        emit_gauge(
            "middleout_engine_enabled",
            "",
            1 if enabled else 0,
            labels={"engine": engine_name},
            emit_meta=False,
        )

    emit_gauge(
        "middleout_jl_similarity_threshold",
        "Configured JL-style near-duplicate similarity threshold (0.0-1.0).",
        f"{settings.jl_similarity_threshold:.6f}",
    )

    lines.append("")  # trailing newline per text-format spec
    return "\n".join(lines)


__all__ = ["render_prometheus"]
