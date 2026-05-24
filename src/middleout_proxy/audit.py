from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from .compression import CompressionAudit
from .config import Settings


LATENCY_BINS_MS: tuple[float, ...] = (
    1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0,
    1000.0, 2500.0, 5000.0, 10000.0, 30000.0, 60000.0, 120000.0,
)


def _bin_index(latency_ms: float) -> int:
    for i, edge in enumerate(LATENCY_BINS_MS):
        if latency_ms <= edge:
            return i
    return len(LATENCY_BINS_MS)


def _quantile_from_hist(counts: list[int], total: int, q: float) -> float:
    if total <= 0:
        return 0.0
    target = max(1, int(math.ceil(q * total)))
    cum = 0
    for i, c in enumerate(counts):
        cum += c
        if cum >= target:
            if i < len(LATENCY_BINS_MS):
                return float(LATENCY_BINS_MS[i])
            return float(LATENCY_BINS_MS[-1] * 2)
    return float(LATENCY_BINS_MS[-1] * 2)


@dataclass
class _Bucket:
    minute_ts: int
    requests: int = 0
    errors: int = 0
    chars_saved_in: int = 0
    chars_saved_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    engines: dict[str, int] = field(default_factory=dict)
    latency_counts: list[int] = field(default_factory=lambda: [0] * (len(LATENCY_BINS_MS) + 1))
    latency_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "minute_ts": self.minute_ts,
            "requests": self.requests,
            "errors": self.errors,
            "chars_saved_in": self.chars_saved_in,
            "chars_saved_out": self.chars_saved_out,
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "engines": dict(self.engines),
            "p50_ms": round(_quantile_from_hist(self.latency_counts, self.latency_total, 0.50), 2),
            "p95_ms": round(_quantile_from_hist(self.latency_counts, self.latency_total, 0.95), 2),
        }


@dataclass
class ProxyStats:
    started_at: float = field(default_factory=time.time)
    requests_total: int = 0
    compressed_requests: int = 0
    chars_saved_in: int = 0
    chars_saved_out: int = 0
    upstream_errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    protected_blocks: int = 0
    bytes_in_total: int = 0
    bytes_out_total: int = 0
    engines_total: dict[str, int] = field(default_factory=dict)
    window_minutes: int = 60
    recent_max: int = 200

    def __post_init__(self) -> None:
        self._buckets: dict[int, _Bucket] = {}
        self._recent: deque = deque(maxlen=max(1, self.recent_max))
        self._latency_global_counts: list[int] = [0] * (len(LATENCY_BINS_MS) + 1)
        self._latency_global_total: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "requests_total": self.requests_total,
            "compressed_requests": self.compressed_requests,
            "chars_saved_in": self.chars_saved_in,
            "chars_saved_out": self.chars_saved_out,
            "upstream_errors": self.upstream_errors,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "protected_blocks": self.protected_blocks,
            "bytes_in_total": self.bytes_in_total,
            "bytes_out_total": self.bytes_out_total,
            "engines_total": dict(self.engines_total),
            "uptime_s": round(time.time() - self.started_at, 3),
            "p50_ms": round(
                _quantile_from_hist(self._latency_global_counts, self._latency_global_total, 0.50),
                2,
            ),
            "p95_ms": round(
                _quantile_from_hist(self._latency_global_counts, self._latency_global_total, 0.95),
                2,
            ),
        }

    def timeseries(self, *, now: float | None = None) -> list[dict[str, Any]]:
        self._evict(now=now)
        return [self._buckets[k].to_dict() for k in sorted(self._buckets.keys())]

    def recent(self, n: int) -> list[dict[str, Any]]:
        if n <= 0:
            return []
        items = list(self._recent)
        return items[-n:]

    def observe(
        self,
        *,
        method: str,
        path: str,
        status_code: int | None,
        chars_saved_in: int,
        chars_saved_out: int,
        engines: dict[str, int],
        latency_ms: float,
        bytes_in: int,
        bytes_out: int,
        request_id: str | None,
        is_error: bool,
        request_audit_summary: dict[str, Any] | None = None,
        response_audit_summary: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> None:
        ts = time.time() if now is None else now
        self.requests_total += 1
        if chars_saved_in or chars_saved_out:
            self.compressed_requests += 1
        self.chars_saved_in += chars_saved_in
        self.chars_saved_out += chars_saved_out
        self.bytes_in_total += max(0, int(bytes_in))
        self.bytes_out_total += max(0, int(bytes_out))
        if is_error:
            self.upstream_errors += 1
        for engine, saved in engines.items():
            if saved <= 0:
                continue
            self.engines_total[engine] = self.engines_total.get(engine, 0) + int(saved)

        idx = _bin_index(max(0.0, float(latency_ms)))
        self._latency_global_counts[idx] += 1
        self._latency_global_total += 1

        bucket = self._bucket_for(ts)
        bucket.requests += 1
        if is_error:
            bucket.errors += 1
        bucket.chars_saved_in += chars_saved_in
        bucket.chars_saved_out += chars_saved_out
        bucket.bytes_in += max(0, int(bytes_in))
        bucket.bytes_out += max(0, int(bytes_out))
        for engine, saved in engines.items():
            if saved <= 0:
                continue
            bucket.engines[engine] = bucket.engines.get(engine, 0) + int(saved)
        bucket.latency_counts[idx] += 1
        bucket.latency_total += 1

        self._recent.append(
            {
                "ts": ts,
                "method": method,
                "path": path,
                "status_code": status_code,
                "ms": round(float(latency_ms), 2),
                "chars_saved_in": chars_saved_in,
                "chars_saved_out": chars_saved_out,
                "bytes_in": int(bytes_in),
                "bytes_out": int(bytes_out),
                "engines": dict(engines),
                "request_id": request_id,
                "is_error": bool(is_error),
                "request_audit": _sanitize_audit_summary(request_audit_summary),
                "response_audit": _sanitize_audit_summary(response_audit_summary),
            }
        )

        self._evict(now=ts)

    def _bucket_for(self, ts: float) -> _Bucket:
        minute_ts = int(ts) - (int(ts) % 60)
        bucket = self._buckets.get(minute_ts)
        if bucket is None:
            bucket = _Bucket(minute_ts=minute_ts)
            self._buckets[minute_ts] = bucket
        return bucket

    def _evict(self, *, now: float | None = None) -> None:
        ts = time.time() if now is None else now
        cutoff = int(ts) - (self.window_minutes * 60)
        stale = [k for k in self._buckets if k <= cutoff]
        for k in stale:
            del self._buckets[k]


def _sanitize_audit_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    if not isinstance(summary, dict):
        return None
    events = summary.get("events")
    if isinstance(events, list):
        sanitized_events: list[dict[str, Any]] = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            clean = {k: v for k, v in ev.items() if k not in {"sample_before", "sample_after"}}
            sanitized_events.append(clean)
        out = dict(summary)
        out["events"] = sanitized_events
        return out
    return dict(summary)


def _audit_to_engine_chars(audit: CompressionAudit | None) -> dict[str, int]:
    if audit is None:
        return {}
    out: dict[str, int] = {}
    for ev in audit.events:
        mode = getattr(ev, "mode", None) or "unknown"
        out[mode] = out.get(mode, 0) + int(getattr(ev, "chars_saved", 0))
    return out


class AuditLogger:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stats = ProxyStats(
            window_minutes=int(getattr(settings, "timeseries_minutes", 60)),
            recent_max=int(getattr(settings, "recent_max", 200)),
        )
        self._lock = Lock()
        self._log_path: Path | None = None
        if settings.audit_enabled:
            settings.audit_log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = settings.audit_log_dir / "audit.jsonl"

    def record(
        self,
        *,
        method: str,
        path: str,
        status_code: int | None,
        request_audit: CompressionAudit,
        response_audit: CompressionAudit | None = None,
        request_id: str | None = None,
        error: str | None = None,
        latency_ms: float | None = None,
        bytes_in: int = 0,
        bytes_out: int = 0,
        now: float | None = None,
    ) -> None:
        chars_saved_in = request_audit.chars_saved if request_audit else 0
        chars_saved_out = response_audit.chars_saved if response_audit else 0
        engines = _audit_to_engine_chars(request_audit)
        for mode, saved in _audit_to_engine_chars(response_audit).items():
            key = mode if mode.endswith("-response") else f"{mode}-response"
            engines[key] = engines.get(key, 0) + saved
        is_error = bool(error) or (status_code is not None and status_code >= 500)

        latency = float(latency_ms) if latency_ms is not None else 0.0
        request_summary = request_audit.to_dict() if request_audit else None
        response_summary = response_audit.to_dict() if response_audit else None

        with self._lock:
            self.stats.cache_hits += request_audit.cache_hits if request_audit else 0
            self.stats.cache_misses += request_audit.cache_misses if request_audit else 0
            self.stats.protected_blocks += (
                request_audit.protected_blocks if request_audit else 0
            )
            if response_audit:
                self.stats.cache_hits += response_audit.cache_hits
                self.stats.cache_misses += response_audit.cache_misses

            self.stats.observe(
                method=method,
                path=path,
                status_code=status_code,
                chars_saved_in=chars_saved_in,
                chars_saved_out=chars_saved_out,
                engines=engines,
                latency_ms=latency,
                bytes_in=bytes_in,
                bytes_out=bytes_out,
                request_id=request_id,
                is_error=is_error,
                request_audit_summary=request_summary,
                response_audit_summary=response_summary,
                now=now,
            )

        if self._log_path:
            entry = {
                "ts": now if now is not None else time.time(),
                "method": method,
                "path": path,
                "status_code": status_code,
                "request_id": request_id,
                "ms": round(latency, 2),
                "bytes_in": int(bytes_in),
                "bytes_out": int(bytes_out),
                "request_compression": request_summary,
                "response_compression": response_summary,
                "error": error,
            }
            with self._lock:
                with self._log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if getattr(self.settings, "log_json", False):
            structured = {
                "ts": _iso_now(now),
                "method": method,
                "path": path,
                "status": status_code,
                "ms": round(latency, 2),
                "chars_saved_input": chars_saved_in,
                "chars_saved_output": chars_saved_out,
                "engines_active": sorted(k for k, v in engines.items() if v > 0),
                "request_id": request_id,
            }
            print(json.dumps(structured, ensure_ascii=False), file=sys.stderr, flush=True)


def _iso_now(now: float | None) -> str:
    t = time.time() if now is None else now
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()

