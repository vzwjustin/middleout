from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

from .adaptive import decide_levels as _adaptive_decide_levels
from .adaptive import should_compress as _adaptive_should_compress
from .audit import AuditLogger
from .budget import UsageBudget
from .cache import CachedResponse, L1Cache, cache_key, canonical_text
from .cache.embedders import HashEmbedder, OpenAIEmbeddingClient
from .cache.l2 import L2Cache
from .cache.vector_stores import InMemoryVectorStore, QdrantVectorStore
from .cache_wall import compute_wall
from .compression import CompressionAudit, PayloadCompressor
from .config import Settings, load_settings
from .cost import (
    CostTracker,
    SSEUsageAccumulator,
    estimate as estimate_cost,
    extract_usage_from_anthropic,
)
from .dashboard import _DASHBOARD_HTML
from .lingua import LinguaCompressor
from .metrics import render_prometheus
from .policies import CompressionPolicy, PolicyRouter
from .preview import preview_compression
from .providers import REGISTRY as PROVIDER_REGISTRY
from .providers import AdapterNotImplemented, select_adapter
from .providers.registry import routes_snapshot as _routes_snapshot
from .rate_limit import RequestLimiter
from .volatile import compress_volatile_tail

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "content-encoding",
}

# Defense in depth: even if upstream ever reflects an auth header, we don't relay it.
RESPONSE_STRIPPED_HEADERS = {
    "authorization",
    "x-api-key",
    "anthropic-api-key",
    "proxy-authorization",
    "set-cookie",
}

settings = load_settings()
compressor = PayloadCompressor(settings)
audit_logger = AuditLogger(settings)
lingua_compressor = LinguaCompressor(
    model_id=settings.lingua_model_id,
    default_ratio=settings.lingua_ratio,
)
l1_cache: L1Cache | None = None
# Always construct the L1 cache so the runtime toggle (`/settings POST
# l1_cache:true`) can flip it on without restarting. If the user disabled
# L1 at startup we still keep the object so the dashboard can flip it.
# A construction failure (bad db_path, FS perms) is logged but does not
# kill the proxy — the runtime toggle simply has no effect.
try:
    l1_cache = L1Cache(
        settings.l1_cache_db_path,
        max_entries=settings.l1_cache_max_entries,
        max_body_bytes=settings.l1_cache_max_body_bytes,
    )
except Exception as e:
    import logging
    logging.getLogger(__name__).warning(
        "L1 cache disabled at startup: %s: %s", type(e).__name__, e
    )
    l1_cache = None

# L2 semantic cache (Phase 2b) — wired into the request pipeline.
# Backend + embedder are picked from settings. Failures during construction
# degrade gracefully: L2 stays off and the misconfig surfaces via /healthz,
# rather than failing the proxy boot.
#
# We ALWAYS try to build a working L2Cache (even if disabled in settings) so
# the runtime toggle in the dashboard can flip it on without a restart. When
# settings say disabled, we start it with `enabled=False` — the runtime flag
# alone flips `.enabled = True` afterwards.
def _build_l2_cache(s: Settings) -> tuple[L2Cache, str | None]:
    try:
        if s.l2_embedder == "openai":
            embedder = OpenAIEmbeddingClient(
                model=s.l2_openai_model,
                dim=s.l2_embedding_dim,
            )
        else:
            embedder = HashEmbedder(dim=s.l2_embedding_dim)
        if s.l2_backend == "qdrant":
            if not s.l2_qdrant_url:
                # Qdrant explicitly requested but not configured — fall back
                # to in-memory so the toggle still works, but surface the
                # misconfig only when the user actually enabled L2.
                if s.l2_cache_enabled:
                    raise ValueError(
                        "BRAIN_L2_BACKEND=qdrant but BRAIN_L2_QDRANT_URL is empty"
                    )
                store = InMemoryVectorStore(max_entries=s.l2_max_entries)
            else:
                store = QdrantVectorStore(
                    url=s.l2_qdrant_url,
                    collection=s.l2_qdrant_collection,
                    dim=s.l2_embedding_dim,
                    api_key=s.l2_qdrant_api_key or None,
                )
        else:
            store = InMemoryVectorStore(max_entries=s.l2_max_entries)
        return (
            L2Cache(
                enabled=bool(s.l2_cache_enabled),
                embedding_client=embedder,
                vector_store=store,
                similarity_threshold=s.l2_similarity_threshold,
            ),
            None,
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "L2 cache disabled at startup: %s: %s", type(e).__name__, e
        )
        return (
            L2Cache(
                embedding_client=None,
                vector_store=None,
                similarity_threshold=s.l2_similarity_threshold,
                enabled=False,
            ),
            f"{type(e).__name__}: {e}",
        )


l2_cache, _l2_init_error = _build_l2_cache(settings)
l2_cache_misconfigured: bool = (
    settings.l2_cache_enabled and not l2_cache.enabled
)

# Cost tracker. Always on — per-request cost is recorded on the response
# path for /v1/messages whenever the upstream returned a JSON body. The
# tracker survives across process lifetime but never persists to disk.
cost_tracker = CostTracker()

# Usage budget. Limits are operator-controlled; default = unlimited so
# the tracker is informational only.
usage_budget = UsageBudget(char_limit=None, token_limit=None)

# Policy router (per-model, per-endpoint overrides). Pulls from
# `MIDDLEOUT_POLICIES` (JSON) at startup. Errors surface as a startup
# failure — invalid policy JSON is operator error, not a runtime fallback.
policy_router = PolicyRouter.from_env()

# Per-client rate limiter. Always-constructed so the runtime toggle can flip
# it on without a restart; the runtime `rate_limit` flag alone gates the
# check. Keyed on a SHA-256 prefix of the Authorization header so raw
# bearer tokens never reach the limiter.
request_limiter = RequestLimiter(
    capacity=settings.rate_limit_capacity,
    refill_per_second=settings.rate_limit_refill_per_second,
)

_CAVEMAN_LEVELS = {"lite", "standard", "aggressive", "ultra"}
_RTK_LEVELS = {"minimal", "standard", "aggressive"}
_JSON_AWARE_LEVELS = {"safe", "standard", "aggressive"}
_LSH_LEVELS = {"conservative", "standard", "aggressive"}
_LINGUA_RATIO_RANGE = (0.05, 0.95)

_RUNTIME_PERSIST_PATH = settings.audit_log_dir / "runtime_settings.json"

# Serializes POST /settings writes so in-flight requests can't observe torn
# nested dicts (e.g., caveman.enabled flipped without level updating). Each
# request takes a snapshot under the lock and uses the snapshot for the rest
# of the handler.
_runtime_lock = asyncio.Lock()

_runtime: dict = {
    "input_compression": settings.input_compression_enabled,
    "output_compression": settings.output_compression_enabled,
    "jl_dedupe": settings.jl_dedupe_enabled,
    "caveman": {"enabled": settings.caveman_enabled, "level": settings.caveman_level},
    "rtk": {"enabled": settings.rtk_enabled, "level": settings.rtk_level},
    "json_aware": {"enabled": settings.json_aware_enabled, "level": settings.json_aware_level},
    "lsh": {"enabled": settings.lsh_enabled, "level": settings.lsh_level},
    "adaptive": settings.adaptive_enabled,
    "lingua": {"enabled": settings.lingua_enabled, "ratio": settings.lingua_ratio},
    "auto_insert_wall": settings.auto_insert_cache_wall,
    "l1_cache": settings.l1_cache_enabled,
    "l2_cache": l2_cache.enabled,
    "rate_limit": settings.rate_limit_enabled,
}

def _load_persisted_runtime() -> None:
    try:
        saved = json.loads(_RUNTIME_PERSIST_PATH.read_text())
        if not isinstance(saved, dict):
            return
        for k in ("input_compression", "output_compression", "jl_dedupe", "auto_insert_wall", "adaptive", "l1_cache", "l2_cache", "rate_limit"):
            if k in saved:
                _runtime[k] = bool(saved[k])
        for engine_key in ("caveman", "rtk", "lingua", "json_aware", "lsh"):
            if engine_key in saved and isinstance(saved[engine_key], dict):
                _runtime[engine_key] = {**_runtime[engine_key], **saved[engine_key]}
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        pass

def _save_runtime() -> None:
    """Atomic write: serialize, write to a sibling tmp file, then rename.

    Power-fail and racing-reader safety: a partial write never replaces the
    canonical file, and a concurrent reader observes either the old contents
    or the new contents — never a mix.
    """
    try:
        _RUNTIME_PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(_runtime)
        tmp = _RUNTIME_PERSIST_PATH.with_suffix(
            _RUNTIME_PERSIST_PATH.suffix + ".tmp"
        )
        tmp.write_text(payload)
        tmp.replace(_RUNTIME_PERSIST_PATH)
    except OSError:
        pass

_load_persisted_runtime()

# Sync the L2 cache's own `.enabled` flag with the runtime snapshot. Without
# this, a process that boots with BRAIN_L2_CACHE_ENABLED unset but a persisted
# `l2_cache: true` runtime flag would report `l2_cache_enabled: True` in
# /settings but `False` in /healthz, and L2 lookups would be silently skipped.
if l2_cache is not None:
    l2_cache.enabled = bool(_runtime.get("l2_cache", False))


def _client_timeout(settings: Settings) -> httpx.Timeout:
    return httpx.Timeout(
        connect=settings.timeout_connect_s,
        read=settings.timeout_read_s,
        write=settings.timeout_connect_s,
        pool=settings.timeout_connect_s,
    )


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: open one shared httpx.AsyncClient for the process.

    The client supports HTTP/2 when the optional ``h2`` dep is installed,
    otherwise falls back to HTTP/1.1. We pin ``follow_redirects=False`` so the
    proxy is the source of truth for redirect handling (Anthropic doesn't 3xx,
    but other providers might). The client is also exposed via ``app.state.http``
    so tests can monkey-patch it.

    Tests that construct the TestClient outside of ``with`` get a pre-seeded
    ``app.state.http = None`` to monkey-patch onto; the real client only opens
    when the lifespan event fires (i.e., inside ``with TestClient(app):``).
    """
    app.state.http = httpx.AsyncClient(
        timeout=_client_timeout(settings),
        follow_redirects=False,
    )
    try:
        yield
    finally:
        # Close every long-lived resource. Each shutdown is independent so a
        # failure in one tier (e.g. SQLite already closed by tests) never
        # prevents another from running.
        try:
            await app.state.http.aclose()
        except Exception:
            pass
        if l1_cache is not None:
            try:
                l1_cache.close()
            except Exception:
                pass


app = FastAPI(
    title="MiddleOut Claude Proxy",
    version="0.2.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

# Pre-seed app.state.http to None so tests that monkey-patch it before
# entering the lifespan context have an attribute to overwrite. Real
# requests always run through `with TestClient(app):` (which triggers the
# lifespan) so this default is overwritten before the first request lands.
app.state.http = None


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "upstream": settings.upstream,
        "input_compression": _runtime["input_compression"],
        "jl_dedupe": _runtime["jl_dedupe"],
        "output_compression": _runtime["output_compression"],
        "caveman_enabled": _runtime["caveman"]["enabled"],
        "rtk_enabled": _runtime["rtk"]["enabled"],
        "json_aware_enabled": _runtime["json_aware"]["enabled"],
        "json_aware_level": _runtime["json_aware"]["level"],
        "lsh_enabled": _runtime["lsh"]["enabled"],
        "lsh_level": _runtime["lsh"]["level"],
        "adaptive_enabled": _runtime["adaptive"],
        "lingua_enabled": _runtime["lingua"]["enabled"],
        "lingua_ratio": _runtime["lingua"]["ratio"],
        "lingua_model_loaded": lingua_compressor.is_loaded,
        "auto_insert_cache_wall": _runtime["auto_insert_wall"],
        "l1_cache_enabled": _runtime["l1_cache"] and l1_cache is not None,
        "l1_cache_backend": settings.l1_cache_db_path if l1_cache is not None else None,
        "l2_cache_enabled": l2_cache.enabled,
        "l2_cache_misconfigured": l2_cache_misconfigured,
        "l2_cache_backend": settings.l2_backend if l2_cache.enabled else None,
        "l2_embedder": settings.l2_embedder if l2_cache.enabled else None,
        "l2_similarity_threshold": settings.l2_similarity_threshold,
        "l2_init_error": _l2_init_error,
        "preserve_anthropic_cache": settings.preserve_anthropic_cache,
        "compression_cache_enabled": settings.compression_cache_enabled,
        "auth_mode": settings.auth_mode,
        "api_key_injection": False,
        "api_key_headers_rejected": True,
        "api_keys_supported": False,
        "providers": sorted(PROVIDER_REGISTRY.keys()),
        "rate_limit_enabled": _runtime["rate_limit"],
        "rate_limit_capacity": request_limiter.capacity,
        "rate_limit_refill_per_second": request_limiter.refill_per_second,
        "phase": "1-cache-aware-compression + 2a-l1-cache + 2b-l2-stub + 3-provider-scaffold + 4-rate-limit+policies",
    }


@app.get("/stats")
async def stats() -> dict[str, Any]:
    snap = audit_logger.stats.snapshot()
    snap["result_cache"] = compressor.result_cache.stats()
    snap["preserve_anthropic_cache"] = settings.preserve_anthropic_cache
    if l1_cache is not None:
        snap["l1_cache"] = l1_cache.stats()
    if l2_cache.enabled:
        snap["l2_cache"] = l2_cache.stats()
        store_stats = getattr(l2_cache.vector_store, "stats", None)
        if callable(store_stats):
            snap["l2_vector_store"] = store_stats()
    return snap


@app.get("/stats/timeseries")
async def stats_timeseries() -> dict[str, Any]:
    """Rolling 1-minute buckets, oldest -> newest, for the last window_minutes.

    Each bucket: minute_ts, requests, errors, chars_saved_in, chars_saved_out,
    bytes_in, bytes_out, engines (per-engine chars_saved), p50_ms, p95_ms.
    Read-only. Never contains raw payload text.
    """
    return {
        "window_minutes": audit_logger.stats.window_minutes,
        "buckets": audit_logger.stats.timeseries(),
    }


@app.get("/stats/recent")
async def stats_recent(n: int = 50) -> dict[str, Any]:
    """Last N audit records (hashes + stats only — NEVER raw payload text)."""
    n = max(0, min(int(n), audit_logger.stats.recent_max))
    return {"count": n, "items": audit_logger.stats.recent(n)}


@app.get("/settings")
async def get_settings() -> dict[str, Any]:
    return dict(_runtime)


def _snapshot_runtime() -> dict[str, Any]:
    """Take a coherent snapshot of `_runtime` for one request to use throughout.

    Cheap dict-of-dicts deep copy — engine settings are tiny so this is well
    under a microsecond. Callers don't hold the lock past the snapshot.
    """
    return {
        "input_compression": _runtime["input_compression"],
        "output_compression": _runtime["output_compression"],
        "jl_dedupe": _runtime["jl_dedupe"],
        "auto_insert_wall": _runtime["auto_insert_wall"],
        "l1_cache": _runtime["l1_cache"],
        "l2_cache": _runtime["l2_cache"],
        "rate_limit": _runtime["rate_limit"],
        "adaptive": _runtime["adaptive"],
        "caveman": dict(_runtime["caveman"]),
        "rtk": dict(_runtime["rtk"]),
        "json_aware": dict(_runtime["json_aware"]),
        "lsh": dict(_runtime["lsh"]),
        "lingua": dict(_runtime["lingua"]),
    }


@app.post("/settings")
async def post_settings(request: Request) -> Response:
    body = await request.json()
    async with _runtime_lock:
        return await _post_settings_locked(body)


async def _post_settings_locked(body: dict) -> Response:
    for k in ("input_compression", "output_compression", "jl_dedupe", "auto_insert_wall", "l1_cache", "l2_cache", "adaptive", "rate_limit"):
        if k in body:
            _runtime[k] = bool(body[k])

    for engine_key, valid_levels in (
        ("caveman", _CAVEMAN_LEVELS),
        ("rtk", _RTK_LEVELS),
        ("json_aware", _JSON_AWARE_LEVELS),
        ("lsh", _LSH_LEVELS),
    ):
        if engine_key not in body:
            continue
        incoming = body[engine_key]
        if not isinstance(incoming, dict):
            return JSONResponse(
                status_code=400,
                content={"error": f"{engine_key} must be an object"},
            )
        current = dict(_runtime[engine_key])
        if "enabled" in incoming:
            current["enabled"] = bool(incoming["enabled"])
        if "level" in incoming:
            level = str(incoming["level"])
            if level not in valid_levels:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"{engine_key} level must be one of "
                        f"{sorted(valid_levels)}, got {level!r}"
                    },
                )
            current["level"] = level
        _runtime[engine_key] = current

    if "lingua" in body:
        incoming = body["lingua"]
        if not isinstance(incoming, dict):
            return JSONResponse(
                status_code=400,
                content={"error": "lingua must be an object"},
            )
        current = dict(_runtime["lingua"])
        if "enabled" in incoming:
            current["enabled"] = bool(incoming["enabled"])
        if "ratio" in incoming:
            try:
                ratio = float(incoming["ratio"])
            except (TypeError, ValueError):
                return JSONResponse(
                    status_code=400,
                    content={"error": "lingua.ratio must be a float"},
                )
            lo, hi = _LINGUA_RATIO_RANGE
            if not lo <= ratio <= hi:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"lingua.ratio must be in [{lo}, {hi}], got {ratio}"},
                )
            current["ratio"] = ratio
        _runtime["lingua"] = current

    _save_runtime()
    # Side-effects: keep L2Cache.enabled in sync with the runtime toggle so
    # the dashboard can flip L2 on/off without a restart. L1 is always-built
    # so its toggle is just the rt["l1_cache"] gate at request time.
    if l2_cache is not None and l2_cache.embedding_client is not None and l2_cache.vector_store is not None:
        l2_cache.enabled = bool(_runtime["l2_cache"])
    return JSONResponse(content=dict(_runtime))




@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    return HTMLResponse(content=_DASHBOARD_HTML)


# --- Brain endpoints (preview, metrics, cost, providers, cache admin) -------


@app.post("/preview")
async def preview(request: Request) -> JSONResponse:
    """Dry-run compression: pass a request payload, get sizes/savings/audit.

    Pure analysis — never touches the network, never writes audit logs. Useful
    for sizing prompts against the cache wall and for evaluating engine
    settings before flipping them in production.
    """
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": f"invalid JSON: {exc.msg}"},
        )
    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=400,
            content={"error": f"body must be a JSON object, got {type(payload).__name__}"},
        )
    async with _runtime_lock:
        rt = _snapshot_runtime()
    try:
        result = preview_compression(
            payload,
            settings,
            jl_dedupe=bool(rt["jl_dedupe"]),
            caveman=rt["caveman"],
            rtk=rt["rtk"],
            input_compression=bool(rt["input_compression"]),
            json_aware=rt["json_aware"],
            lsh=rt["lsh"],
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"preview failed: {type(exc).__name__}: {exc}",
            },
        )
    return JSONResponse(content=result)


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    """Prometheus-format snapshot of audit + cache + cost counters.

    The per-engine gauges report the live runtime state (input/output
    compression, jl_dedupe, caveman, rtk, etc.) — not the startup config —
    so an operator who flips a knob via /settings sees the change reflected
    in Prometheus immediately.
    """
    snap = audit_logger.stats.snapshot()
    snap["result_cache"] = compressor.result_cache.stats()
    async with _runtime_lock:
        runtime_snapshot = _snapshot_runtime()
    body = render_prometheus(snap, settings=settings, runtime=runtime_snapshot)
    return PlainTextResponse(content=body, media_type="text/plain; version=0.0.4")


@app.get("/cost")
async def cost() -> dict[str, Any]:
    """Cumulative cost snapshot: USD by model, total requests, unmatched count."""
    snap = cost_tracker.snapshot()
    snap["budget"] = usage_budget.snapshot()
    return snap


@app.post("/cost/reset")
async def cost_reset() -> dict[str, Any]:
    """Zero the cost tracker counters AND the usage budget counters.

    Operators rolling over their billing period typically want both axes
    cleared atomically — otherwise `/cost` keeps showing stale
    ``budget.chars_used`` values that no longer match the per-model rows.
    """
    cost_tracker.reset()
    usage_budget.reset()
    return {"reset": True, "total_usd": 0.0, "budget_reset": True}


@app.post("/budget/reset")
async def budget_reset() -> dict[str, Any]:
    """Zero just the usage budget counters. Cost tracker is left untouched."""
    usage_budget.reset()
    return {"reset": True, "budget": usage_budget.snapshot()}


@app.get("/providers")
async def providers() -> dict[str, Any]:
    """Snapshot of registered provider adapters + routing rules."""
    return _routes_snapshot()


@app.get("/cache/stats")
async def cache_stats() -> dict[str, Any]:
    """Current L1 + L2 cache state for the operator."""
    l1_stats: dict[str, Any] = {"enabled": False}
    if l1_cache is not None and _runtime["l1_cache"]:
        try:
            l1_stats = l1_cache.stats()  # type: ignore[union-attr]
            l1_stats["enabled"] = True
        except Exception as exc:
            l1_stats = {"enabled": True, "error": str(exc)}
    return {
        "l1": l1_stats,
        "l2": l2_cache.stats(),
        "l2_misconfigured": l2_cache_misconfigured,
    }


@app.post("/cache/purge")
async def cache_purge() -> dict[str, Any]:
    """Drop every L1 + L2 entry. Returns per-tier cleared counts.

    L1: SQLite (or memory-backed) exact-match cache. L2: in-process
    InMemoryVectorStore or remote QdrantVectorStore via the L2Cache wrapper.
    Both purges are best-effort; one tier's failure never blocks the other.
    """
    l1_cleared = 0
    if l1_cache is not None:
        try:
            l1_cleared = l1_cache.purge()  # type: ignore[union-attr]
        except Exception:
            l1_cleared = 0
    l2_cleared = 0
    try:
        l2_cleared = l2_cache.clear()
    except Exception:
        l2_cleared = 0
    return {"l1_cleared": l1_cleared, "l2_cleared": l2_cleared}


@app.get("/budget")
async def budget() -> dict[str, Any]:
    """Process-level cumulative usage + configured limits."""
    return usage_budget.snapshot()


@app.get("/rate-limit")
async def rate_limit() -> dict[str, Any]:
    """Per-client token-bucket state. Buckets are keyed on a hash of each
    client's Authorization header; raw tokens never appear here. Tracking is
    informational regardless of the enabled toggle — the toggle only gates
    enforcement at request time.
    """
    snap = request_limiter.stats()
    snap["enabled"] = _runtime["rate_limit"]
    snap["policies_rules"] = len(policy_router.rules)
    snap["policies_default_is_vanilla"] = (
        policy_router.default == CompressionPolicy()
    )
    return snap


@app.get("/policies")
async def policies_route() -> dict[str, Any]:
    """Current per-(model, endpoint) compression policy router config."""
    rules_out = []
    for r in policy_router.rules:
        rules_out.append({
            "model_glob": r.model_glob,
            "endpoint": r.endpoint,
            "policy": {
                "input_compression": r.policy.input_compression,
                "output_compression": r.policy.output_compression,
                "jl_dedupe": r.policy.jl_dedupe,
                "caveman_enabled": r.policy.caveman_enabled,
                "caveman_level": r.policy.caveman_level,
                "rtk_enabled": r.policy.rtk_enabled,
                "rtk_level": r.policy.rtk_level,
                "max_text_chars": r.policy.max_text_chars,
            },
        })
    return {
        "rules": rules_out,
        "default": {
            "input_compression": policy_router.default.input_compression,
            "output_compression": policy_router.default.output_compression,
            "jl_dedupe": policy_router.default.jl_dedupe,
            "caveman_enabled": policy_router.default.caveman_enabled,
            "caveman_level": policy_router.default.caveman_level,
            "rtk_enabled": policy_router.default.rtk_enabled,
            "rtk_level": policy_router.default.rtk_level,
            "max_text_chars": policy_router.default.max_text_chars,
        },
    }


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request) -> Response:
    # Local endpoints are handled above, but keep a guard for accidental empty path.
    if path in {"", "/"}:
        return JSONResponse(
            {
                "name": "middleout-claude-proxy",
                "health": "/healthz",
                "stats": "/stats",
                "anthropic_messages": "/v1/messages",
            }
        )

    try:
        request_headers = _forward_request_headers(request.headers, settings)
    except StrictSubscriptionAuthError as exc:
        return JSONResponse(
            status_code=401,
            content={
                "type": "error",
                "error": {
                    "type": "strict_subscription_auth_error",
                    "message": str(exc),
                },
            },
        )

    # X-Brain-Model-Hint: lets a client pin a request to a specific provider
    # adapter without changing the body's `model` field. When the hint
    # resolves to a not-yet-implemented adapter we fail loud with 501 so the
    # operator notices instead of getting a confusing upstream error.
    model_hint = request.headers.get("x-brain-model-hint")
    if model_hint:
        try:
            adapter = select_adapter(model="", model_hint=model_hint)
        except Exception:
            adapter = None
        if adapter is not None and adapter.name != "anthropic":
            # Probe the adapter early; if it raises AdapterNotImplemented we
            # short-circuit with 501. The IR doesn't matter because the
            # adapter rejects regardless of payload.
            from .providers.base import RequestIR

            probe = RequestIR(payload={"model": ""}, headers={}, endpoint=path)
            try:
                adapter.translate_request(probe)
            except AdapterNotImplemented as exc:
                return JSONResponse(
                    status_code=501,
                    content={
                        "type": "error",
                        "error": {
                            "type": "adapter_not_implemented",
                            "message": str(exc),
                            "adapter": adapter.name,
                        },
                    },
                )
            # If translate_request did NOT raise, the operator wired a real
            # adapter — but the integration glue for routing to non-anthropic
            # upstreams ships in a follow-up. Fall through to anthropic for now.
            request_headers["x-brain-resolved-adapter"] = adapter.name

    upstream_url = f"{settings.upstream}/{path}"
    if request.url.query:
        upstream_url = f"{upstream_url}?{request.url.query}"

    body_bytes = await request.body()
    started_perf = time.perf_counter()
    bytes_in = len(body_bytes) if body_bytes else 0
    request_audit = CompressionAudit(endpoint=path)
    outgoing_content: bytes | None = body_bytes if body_bytes else None
    wall_auto_inserted = False
    lingua_chars_saved = 0
    # `cache_lookup_payload` is the payload we'll key L1 cache on — set when
    # we have a JSON-shaped /v1/messages request. Captured AFTER compression
    # so two requests that compress to the same bytes share a cache entry.
    cache_lookup_payload: dict[str, Any] | None = None
    # `request_model` is the model id from the request body (Anthropic Messages
    # API). Used for audit-log tagging so operators can filter by model later.
    request_model: str | None = None

    # FIX A: snapshot _runtime once under the lock. All subsequent reads in
    # this handler use the snapshot, so a concurrent POST /settings cannot
    # tear a nested dict mid-compression.
    async with _runtime_lock:
        rt = _snapshot_runtime()

    # RATE LIMIT (Phase 4). Per-client token bucket keyed on a SHA-256 prefix
    # of the Authorization header. Raw bearer tokens are never persisted or
    # logged — the limiter only sees the hash. Off by default; flip via
    # /settings POST {"rate_limit": true} or BRAIN_RATE_LIMIT_ENABLED=1.
    if rt["rate_limit"]:
        auth_header = request.headers.get("authorization") or ""
        if auth_header:
            client_key = hashlib.sha256(auth_header.encode("utf-8")).hexdigest()[:16]
            try:
                allowed = await request_limiter.check(client_key)
            except Exception:
                allowed = True  # never break a real request on a limiter bug
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    headers={
                        "retry-after": "1",
                        "x-brain-rate-limit": "exceeded",
                    },
                    content={
                        "type": "error",
                        "error": {
                            "type": "rate_limit_error",
                            "message": (
                                f"client exceeded local rate limit "
                                f"({request_limiter.capacity} req burst, "
                                f"{request_limiter.refill_per_second}/s sustained)"
                            ),
                        },
                    },
                )

    try:
        if rt["input_compression"] and _should_transform_json_request(path, request.method, request.headers, body_bytes):
            payload = json.loads(body_bytes.decode("utf-8"))
            if isinstance(payload, dict):
                m = payload.get("model")
                if isinstance(m, str):
                    request_model = m

            # POLICY ROUTER: per-(model, endpoint) overrides for compression
            # knobs. Only applied when MIDDLEOUT_POLICIES configured a real
            # ruleset — an empty router with a vanilla default is a no-op so
            # operators who never set a policy keep their runtime-toggle behavior.
            if policy_router.rules or policy_router.default != CompressionPolicy():
                resolved = policy_router.resolve(
                    model=request_model, endpoint=path
                )
                rt["input_compression"] = resolved.input_compression
                rt["output_compression"] = resolved.output_compression
                rt["jl_dedupe"] = resolved.jl_dedupe
                rt["caveman"] = {
                    "enabled": resolved.caveman_enabled,
                    "level": resolved.caveman_level,
                }
                rt["rtk"] = {
                    "enabled": resolved.rtk_enabled,
                    "level": resolved.rtk_level,
                }
                # If policy turned off input_compression mid-flight, we still
                # captured request_model above. Short-circuit the rest of the
                # compression pipeline by jumping into _AdaptiveSkip semantics.
                if not rt["input_compression"]:
                    outgoing_content = body_bytes
                    cache_lookup_payload = payload
                    raise _AdaptiveSkip()

            # Phase 1: LLMLingua-2 on the volatile tail, gated by the cache wall.
            # This runs BEFORE the legacy middle-out engines so they see the
            # already-shrunk tail. Both layers honor cache_control.
            lingua_cfg = rt["lingua"]
            if lingua_cfg["enabled"]:
                wall = compute_wall(
                    payload, auto_insert=bool(rt["auto_insert_wall"])
                )
                wall_auto_inserted = wall.auto_inserted
                payload, volatile_audit = compress_volatile_tail(
                    payload,
                    wall=wall,
                    lingua=lingua_compressor,
                    ratio=lingua_cfg["ratio"],
                    deepcopy_payload=False,  # already a fresh json.loads copy
                )
                lingua_chars_saved = volatile_audit.chars_saved

            # ADAPTIVE POLICY: if enabled, the policy picks engine levels based
            # on the payload's size and the target model's context window. The
            # policy can also short-circuit compression entirely on very small
            # payloads. Caller-supplied levels in `rt` are used as fallbacks
            # when the policy returns a value we don't know how to handle.
            engine_caveman = rt["caveman"]
            engine_rtk = rt["rtk"]
            engine_json_aware = rt["json_aware"]
            engine_lsh = rt["lsh"]
            engine_jl = rt["jl_dedupe"]
            if rt["adaptive"]:
                if not _adaptive_should_compress(payload):
                    # Too small to bother. Send the body unchanged and skip the
                    # rest of the compression pipeline.
                    outgoing_content = body_bytes
                    cache_lookup_payload = payload
                    raise _AdaptiveSkip()
                levels = _adaptive_decide_levels(payload)
                if levels.get("middle_out") == "off":
                    # Disable middle-out via the existing min_omission threshold
                    # path is too intrusive; instead just route through unchanged.
                    pass  # middle-out level is set per-call via settings; keep default
                cav_level = levels.get("caveman")
                if cav_level in _CAVEMAN_LEVELS:
                    engine_caveman = {"enabled": True, "level": cav_level}
                rtk_level = levels.get("rtk")
                if rtk_level in _RTK_LEVELS:
                    engine_rtk = {"enabled": True, "level": rtk_level}
                ja_level = levels.get("json_aware")
                if ja_level in _JSON_AWARE_LEVELS:
                    engine_json_aware = {"enabled": True, "level": ja_level}
                lsh_level = levels.get("lsh")
                if lsh_level in _LSH_LEVELS:
                    engine_lsh = {"enabled": True, "level": lsh_level}
                if "jl_dedupe" in levels:
                    engine_jl = bool(levels["jl_dedupe"])
            transformed, request_audit = compressor.compress_request_payload(
                payload,
                endpoint=path,
                jl_dedupe=engine_jl,
                caveman=engine_caveman,
                rtk=engine_rtk,
                json_aware=engine_json_aware,
                lsh=engine_lsh,
                force_enabled=True,
            )
            # FIX D: when nothing was touched, send the original bytes. Anthropic's
            # prompt cache is byte-keyed; re-encoding with stable separators still
            # produces different bytes than the client's encoder (key order,
            # spaces, escaping) and would invalidate the upstream cache.
            any_change = (
                request_audit.touched
                or lingua_chars_saved > 0
                or wall_auto_inserted
            )
            if any_change:
                outgoing_content = json.dumps(
                    transformed, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")
                request_headers["content-type"] = "application/json"
            # else: outgoing_content stays as body_bytes — byte-identical passthrough
            cache_lookup_payload = transformed
    except _AdaptiveSkip:
        # Adaptive policy decided the payload is too small to benefit from
        # compression; outgoing_content + cache_lookup_payload were set above.
        request_audit.events.clear()
    except Exception as exc:  # Keep the proxy useful even if compression fails.
        request_audit.events.clear()
        request_headers["x-middleout-warning"] = f"compression skipped: {type(exc).__name__}"
        outgoing_content = body_bytes if body_bytes else None
        lingua_chars_saved = 0
        wall_auto_inserted = False

    method = request.method.upper()

    # L1 EXACT-MATCH CACHE LOOKUP (non-streaming only).
    # The key is computed on the COMPRESSED payload — two clients that compress
    # to the same bytes share the cache. Streaming requests are not cached this
    # phase: SSE chunk-boundary replay deserves its own implementation.
    l1_status: str | None = None
    l1_key: str | None = None
    l2_status: str | None = None
    l2_similarity: float | None = None
    cacheable = (
        cache_lookup_payload is not None
        and method == "POST"
        and path.strip("/") == "v1/messages"
        and not cache_lookup_payload.get("stream", False)
    )
    cache_active = (
        cacheable
        and rt["l1_cache"]
        and l1_cache is not None
    )
    l2_active = (
        cacheable
        and rt["l2_cache"]
        and l2_cache.enabled
    )
    if cache_active:
        try:
            l1_key = cache_key(cache_lookup_payload)  # type: ignore[arg-type]
            cached = l1_cache.get(l1_key)  # type: ignore[union-attr]
        except Exception:
            cached = None
        if cached is not None:
            l1_status = "hit"
            cached_headers = dict(cached.headers)
            cached_headers["x-brain-l1-cache"] = "hit"
            cached_headers["x-brain-l1-hit-count"] = str(cached.hit_count)
            cached_headers.update(
                _brain_headers(
                    lingua_chars_saved=lingua_chars_saved,
                    wall_auto_inserted=wall_auto_inserted,
                )
            )
            audit_logger.record(
                method=method,
                path=path,
                status_code=cached.status_code,
                request_audit=request_audit,
                request_id=cached_headers.get("request-id"),
                latency_ms=(time.perf_counter() - started_perf) * 1000.0,
                bytes_in=bytes_in,
                bytes_out=len(cached.body),
                model=request_model,
            )
            return Response(
                content=cached.body,
                status_code=cached.status_code,
                headers=cached_headers,
                media_type=cached.media_type,
            )
        l1_status = "miss"

    # L2 SEMANTIC CACHE LOOKUP on L1 miss. Embed the same canonical normalized
    # JSON text that L1 hashed (keeps the two layers aligned on what "identical"
    # means). The L2 hit is served byte-for-byte, same as L1.
    if l2_active and (l1_status in (None, "miss")):
        try:
            embed_text = canonical_text(cache_lookup_payload)  # type: ignore[arg-type]
            hit = l2_cache.get_similar(embed_text)
        except Exception:
            hit = None
        if hit is not None:
            l2_status = "hit"
            l2_similarity = float(hit.similarity)
            cached_headers = dict(hit.response.headers)
            cached_headers["x-brain-l2-cache"] = "hit"
            cached_headers["x-brain-l2-similarity"] = f"{l2_similarity:.4f}"
            if l1_status is not None:
                cached_headers["x-brain-l1-cache"] = l1_status
            cached_headers.update(
                _brain_headers(
                    lingua_chars_saved=lingua_chars_saved,
                    wall_auto_inserted=wall_auto_inserted,
                )
            )
            audit_logger.record(
                method=method,
                path=path,
                status_code=hit.response.status_code,
                request_audit=request_audit,
                request_id=cached_headers.get("request-id"),
                latency_ms=(time.perf_counter() - started_perf) * 1000.0,
                bytes_in=bytes_in,
                bytes_out=len(hit.response.body),
                model=request_model,
            )
            return Response(
                content=hit.response.body,
                status_code=hit.response.status_code,
                headers=cached_headers,
                media_type=hit.response.media_type,
            )
        l2_status = "miss"

    try:
        if _is_streaming_messages(path, outgoing_content):
            return await _streaming_forward(
                method=method,
                upstream_url=upstream_url,
                headers=request_headers,
                content=outgoing_content,
                request_audit=request_audit,
                path=path,
                lingua_chars_saved=lingua_chars_saved,
                wall_auto_inserted=wall_auto_inserted,
                started_perf=started_perf,
                bytes_in=bytes_in,
                request_model=request_model,
            )

        upstream_response = await app.state.http.request(
            method, upstream_url, headers=request_headers, content=outgoing_content
        )
        response_headers = _forward_response_headers(upstream_response.headers)
        response_headers.update(_compression_headers(request_audit, prefix="input"))
        response_headers.update(
            _brain_headers(
                lingua_chars_saved=lingua_chars_saved,
                wall_auto_inserted=wall_auto_inserted,
            )
        )
        if l1_status is not None:
            response_headers["x-brain-l1-cache"] = l1_status
        if l2_status is not None:
            response_headers["x-brain-l2-cache"] = l2_status

        response_content = upstream_response.content
        response_audit: CompressionAudit | None = None
        if rt["output_compression"] and _should_transform_json_response(path, upstream_response.headers, response_content):
            try:
                response_payload = upstream_response.json()
                transformed_response, response_audit = compressor.compress_response_payload(
                    response_payload, endpoint=path, force_enabled=True,
                )
                if response_audit.touched:
                    response_content = json.dumps(
                        transformed_response, separators=(",", ":"), ensure_ascii=False
                    ).encode("utf-8")
                    response_headers.update(_compression_headers(response_audit, prefix="output"))
                    response_headers["content-type"] = "application/json"
            except Exception:
                response_audit = None

        # COST TRACKING. Anthropic's usage block lives at the top level of a
        # successful Messages response. We parse it once, look up the price,
        # record to the tracker, and stamp x-brain-cost-usd on the outbound
        # response. Failures here never propagate — cost is informational.
        try:
            if path.strip("/") == "v1/messages" and 200 <= upstream_response.status_code < 300:
                response_payload_for_cost: dict[str, Any] | None = None
                ct_lower = upstream_response.headers.get("content-type", "").lower()
                if "application/json" in ct_lower and response_content:
                    try:
                        response_payload_for_cost = json.loads(response_content.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        response_payload_for_cost = None
                if isinstance(response_payload_for_cost, dict):
                    model_id = response_payload_for_cost.get("model")
                    if not isinstance(model_id, str) or not model_id:
                        # Fall back to the request body's model field — useful
                        # when an adapter rewrote the response payload.
                        try:
                            req_payload = (
                                cache_lookup_payload
                                if cache_lookup_payload is not None
                                else json.loads(body_bytes.decode("utf-8"))
                            )
                            if isinstance(req_payload, dict):
                                model_id = str(req_payload.get("model") or "")
                        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                            model_id = ""
                    usage = extract_usage_from_anthropic(response_payload_for_cost)
                    cost_record = _record_anthropic_cost(
                        model=model_id or "",
                        usage=usage,
                        bytes_in=bytes_in,
                    )
                    if cost_record.matched:
                        response_headers["x-brain-cost-usd"] = (
                            f"{cost_record.usd:.6f}"
                        )
        except Exception:
            pass

        audit_logger.record(
            method=method,
            path=path,
            status_code=upstream_response.status_code,
            request_audit=request_audit,
            response_audit=response_audit,
            request_id=upstream_response.headers.get("request-id"),
            latency_ms=(time.perf_counter() - started_perf) * 1000.0,
            bytes_in=bytes_in,
            bytes_out=len(response_content) if response_content else 0,
            model=request_model,
        )

        # L1 + L2 CACHE STORE: on a successful upstream response, populate
        # both layers. L1 stores by exact-match key; L2 stores by embedding of
        # the same canonical normalized JSON. Both put() calls are best-effort.
        if (cache_active or l2_active) and 200 <= upstream_response.status_code < 300 and cache_lookup_payload is not None:
            cached_resp = CachedResponse(
                status_code=upstream_response.status_code,
                headers=dict(response_headers),
                body=bytes(response_content) if response_content else b"",
                media_type=upstream_response.headers.get("content-type"),
            )
            if cache_active and l1_key is not None:
                try:
                    l1_cache.put(l1_key, cached_resp)  # type: ignore[union-attr]
                except Exception:
                    pass
            if l2_active:
                try:
                    # The L2 point_id mirrors the L1 key when available so the
                    # two layers share a stable identity for the same content.
                    point_id = l1_key or cache_key(cache_lookup_payload)
                    embed_text = canonical_text(cache_lookup_payload)
                    l2_cache.put_similar(embed_text, cached_resp, point_id=point_id)
                except Exception:
                    pass

        return Response(
            content=response_content,
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=upstream_response.headers.get("content-type"),
        )
    except Exception as exc:
        audit_logger.record(
            method=method,
            path=path,
            status_code=None,
            request_audit=request_audit,
            error=f"{type(exc).__name__}: {exc}",
            latency_ms=(time.perf_counter() - started_perf) * 1000.0,
            bytes_in=bytes_in,
            bytes_out=0,
            model=request_model,
        )
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {
                    "type": "proxy_upstream_error",
                    "message": f"MiddleOut proxy could not reach upstream: {type(exc).__name__}: {exc}",
                },
            },
        )


async def _streaming_forward(
    *,
    method: str,
    upstream_url: str,
    headers: dict[str, str],
    content: bytes | None,
    request_audit: CompressionAudit,
    path: str,
    started_perf: float,
    bytes_in: int,
    lingua_chars_saved: int = 0,
    wall_auto_inserted: bool = False,
    request_model: str | None = None,
) -> StreamingResponse:
    # FIX B: pin accept-encoding to identity for streaming. httpx otherwise
    # injects "gzip, deflate, br" by default; if upstream returns a non-SSE
    # gzipped error body (auth failure, 5xx), `aiter_raw()` would yield raw
    # gzip bytes that the SSE client can't decode.
    headers = dict(headers)
    headers["accept-encoding"] = "identity"
    req = app.state.http.build_request(method, upstream_url, headers=headers, content=content)
    upstream_response = await app.state.http.send(req, stream=True)
    response_headers = _forward_response_headers(upstream_response.headers)
    response_headers.update(_compression_headers(request_audit, prefix="input"))
    response_headers.update(
        _brain_headers(
            lingua_chars_saved=lingua_chars_saved,
            wall_auto_inserted=wall_auto_inserted,
        )
    )

    bytes_out = 0
    # SSE cost tracking: only attach an accumulator on successful SSE responses
    # to the Messages API. Non-SSE error bodies, non-2xx, and non-Messages paths
    # all skip the parse — the accumulator costs zero in those cases.
    upstream_ct = upstream_response.headers.get("content-type", "").lower()
    sse_acc: SSEUsageAccumulator | None = None
    if (
        path.strip("/") == "v1/messages"
        and 200 <= upstream_response.status_code < 300
        and "text/event-stream" in upstream_ct
    ):
        sse_acc = SSEUsageAccumulator()

    async def body_iter():
        nonlocal bytes_out
        try:
            async for chunk in upstream_response.aiter_raw():
                bytes_out += len(chunk)
                if sse_acc is not None:
                    sse_acc.feed(chunk)
                yield chunk
        finally:
            # Record cost from the SSE stream if we got at least the initial
            # message_start event. Done in `finally` so client disconnects
            # mid-stream still credit whatever we already parsed.
            if sse_acc is not None and sse_acc.saw_message_start:
                try:
                    _record_anthropic_cost(
                        model=sse_acc.model or request_model or "",
                        usage=sse_acc.snapshot(),
                        bytes_in=bytes_in,
                    )
                except Exception:  # noqa: BLE001 — cost is informational
                    pass
            audit_logger.record(
                method=method,
                path=path,
                status_code=upstream_response.status_code,
                request_audit=request_audit,
                request_id=upstream_response.headers.get("request-id"),
                latency_ms=(time.perf_counter() - started_perf) * 1000.0,
                bytes_in=bytes_in,
                bytes_out=bytes_out,
                model=request_model,
            )
            await upstream_response.aclose()

    return StreamingResponse(
        body_iter(),
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type=upstream_response.headers.get("content-type", "text/event-stream"),
    )


class _AdaptiveSkip(Exception):
    """Raised internally by the adaptive policy to short-circuit compression.

    Caught by the request handler's catch-all; the payload is sent unchanged.
    Never escapes the handler.
    """


class StrictSubscriptionAuthError(ValueError):
    """Raised when a request is not using Claude Code subscription/OAuth passthrough."""


def _forward_request_headers(headers: Any, settings: Settings) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    saw_api_key_header = False

    for key, value in headers.items():
        lower = key.lower()
        if lower in HOP_BY_HOP_HEADERS:
            continue
        # Strict subscription mode: API-key headers are never forwarded upstream.
        if lower in {"x-api-key", "anthropic-api-key"}:
            saw_api_key_header = True
            continue
        # Let httpx manage decompression/recompression behavior cleanly.
        if lower == "accept-encoding":
            continue
        forwarded[lower] = value

    if saw_api_key_header:
        raise StrictSubscriptionAuthError(
            "This build is subscription/OAuth-only and rejected an X-Api-Key style header. "
            "Unset ANTHROPIC_API_KEY, remove apiKeyHelper output, and run Claude Code /status "
            "until it shows your Claude subscription login."
        )

    authorization = forwarded.get("authorization", "").strip()
    if not authorization.lower().startswith("bearer "):
        raise StrictSubscriptionAuthError(
            "Missing OAuth Authorization: Bearer header from Claude Code. Start Claude Code with "
            "ANTHROPIC_BASE_URL=http://127.0.0.1:8787 after logging in with /login, and make sure "
            "ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN are unset in the Claude Code shell."
        )
    # OAuth bearer tokens per RFC 6750 never contain commas. A comma-folded
    # Authorization value like `Bearer good, ApiKey sk-...` would otherwise pass
    # the startswith check above — reject it.
    if "," in authorization:
        raise StrictSubscriptionAuthError(
            "Authorization header contains a comma; comma-folded credentials "
            "are rejected to prevent smuggling API keys past the Bearer check."
        )

    if "anthropic-version" not in forwarded:
        forwarded["anthropic-version"] = settings.default_anthropic_version

    forwarded["x-middleout-proxy"] = "middleout-claude-proxy/0.2.0-strict-subscription"
    return forwarded


def _forward_response_headers(headers: Any) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    for key, value in headers.items():
        lower = key.lower()
        if lower in HOP_BY_HOP_HEADERS:
            continue
        # Defense in depth: never echo auth-leaking headers from upstream back
        # to the client, even if Anthropic ever reflects them in an error body.
        if lower in RESPONSE_STRIPPED_HEADERS:
            continue
        # We may alter body bytes, so content-length is intentionally omitted above.
        forwarded[lower] = value
    return forwarded


def _compression_headers(audit: CompressionAudit, *, prefix: str) -> dict[str, str]:
    # Emit both the legacy `x-middleout-*` and the new `x-brain-*` names so
    # operators and dashboards can pick whichever prefix they prefer.
    saved = str(audit.chars_saved)
    events = str(len(audit.events))
    headers: dict[str, str] = {
        f"x-middleout-{prefix}-chars-saved": saved,
        f"x-middleout-{prefix}-events": events,
        # `prefix` is "input" or "output"; map to in/out for the brain alias.
        f"x-brain-chars-saved-{'in' if prefix == 'input' else 'out'}": saved,
    }
    # x-brain-engines is a comma-separated list of engines that actually fired
    # (i.e. saved > 0). Easier to grep than digging the structured audit.
    by_engine: dict[str, int] = {}
    for ev in audit.events:
        mode = getattr(ev, "mode", None) or "unknown"
        if int(getattr(ev, "chars_saved", 0)) <= 0:
            continue
        by_engine[mode] = by_engine.get(mode, 0) + int(ev.chars_saved)
    if by_engine:
        # Sort by chars-saved descending; helps operators scan top engines first.
        ordered = sorted(by_engine.items(), key=lambda kv: -kv[1])
        headers["x-brain-engines"] = ",".join(f"{k}={v}" for k, v in ordered)
    return headers


def _brain_headers(*, lingua_chars_saved: int, wall_auto_inserted: bool) -> dict[str, str]:
    out: dict[str, str] = {}
    if lingua_chars_saved > 0:
        out["x-brain-lingua-chars-saved"] = str(lingua_chars_saved)
    if wall_auto_inserted:
        out["x-brain-wall-inserted"] = "1"
    return out


def _record_anthropic_cost(
    *,
    model: str,
    usage: dict[str, int],
    bytes_in: int,
) -> Any:
    """Record cost + budget for a parsed Anthropic usage block.

    Returns the :class:`CostRecord` so the caller can stamp the response
    header when running in non-streaming mode. Never raises — cost is
    informational.
    """
    cost_record = estimate_cost(
        provider="anthropic",
        model=model or "",
        **usage,
    )
    try:
        cost_tracker.record(cost_record)
    except Exception:  # noqa: BLE001 — informational pipeline must not break the request
        pass
    try:
        usage_budget.record(
            chars=int(bytes_in or 0),
            tokens=int(
                usage.get("input_tokens", 0)
                + usage.get("output_tokens", 0)
                + usage.get("cache_write_tokens", 0)
            ),
        )
    except (TypeError, ValueError):
        pass
    return cost_record


def _should_transform_json_request(path: str, method: str, headers: Any, body: bytes) -> bool:
    if method.upper() != "POST" or not body:
        return False
    normalized = path.strip("/")
    if normalized not in {"v1/messages", "v1/messages/count_tokens"}:
        return False
    content_type = headers.get("content-type", "")
    if "application/json" in content_type.lower():
        return True
    # Body may start with whitespace before the opening brace; lstrip catches that.
    return body.lstrip()[:1] in {b"{", b"["}


def _should_transform_json_response(path: str, headers: Any, body: bytes) -> bool:
    # Pure routing predicate: does the response look like one we *could*
    # rewrite? The runtime decision (`rt["output_compression"]`) is enforced
    # at the call site. We deliberately do NOT consult
    # `settings.output_compression_enabled` here — that would lock the gate
    # to the startup value and silently break the runtime toggle.
    if not body:
        return False
    if path.strip("/") != "v1/messages":
        return False
    content_type = headers.get("content-type", "")
    return "application/json" in content_type.lower()


def _is_streaming_messages(path: str, body: bytes | None) -> bool:
    if path.strip("/") != "v1/messages" or not body:
        return False
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return False
    return bool(payload.get("stream"))


def main() -> None:
    uvicorn.run(
        "middleout_proxy.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
