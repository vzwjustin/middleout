from __future__ import annotations

import json
import time
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .audit import AuditLogger
from .compression import CompressionAudit, PayloadCompressor
from .config import Settings, load_settings
from .dashboard import _DASHBOARD_HTML

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

settings = load_settings()
compressor = PayloadCompressor(settings)
audit_logger = AuditLogger(settings)

_CAVEMAN_LEVELS = {"lite", "standard", "aggressive", "ultra"}
_RTK_LEVELS = {"minimal", "standard", "aggressive"}

_RUNTIME_PERSIST_PATH = settings.audit_log_dir / "runtime_settings.json"

_runtime: dict = {
    "input_compression": settings.input_compression_enabled,
    "output_compression": settings.output_compression_enabled,
    "jl_dedupe": settings.jl_dedupe_enabled,
    "caveman": {"enabled": settings.caveman_enabled, "level": settings.caveman_level},
    "rtk": {"enabled": settings.rtk_enabled, "level": settings.rtk_level},
}

def _load_persisted_runtime() -> None:
    try:
        saved = json.loads(_RUNTIME_PERSIST_PATH.read_text())
        for k in ("input_compression", "output_compression", "jl_dedupe"):
            if k in saved:
                _runtime[k] = bool(saved[k])
        for engine_key in ("caveman", "rtk"):
            if engine_key in saved and isinstance(saved[engine_key], dict):
                _runtime[engine_key] = {**_runtime[engine_key], **saved[engine_key]}
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

def _save_runtime() -> None:
    try:
        _RUNTIME_PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        _RUNTIME_PERSIST_PATH.write_text(json.dumps(_runtime))
    except OSError:
        pass

_load_persisted_runtime()

app = FastAPI(
    title="MiddleOut Claude Proxy",
    version="0.2.0",
    docs_url="/docs",
    redoc_url=None,
)


def _client_timeout(settings: Settings) -> httpx.Timeout:
    return httpx.Timeout(
        connect=settings.timeout_connect_s,
        read=settings.timeout_read_s,
        write=settings.timeout_connect_s,
        pool=settings.timeout_connect_s,
    )


@app.on_event("startup")
async def startup() -> None:
    app.state.http = httpx.AsyncClient(timeout=_client_timeout(settings), follow_redirects=False)


@app.on_event("shutdown")
async def shutdown() -> None:
    await app.state.http.aclose()


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
        "preserve_anthropic_cache": settings.preserve_anthropic_cache,
        "compression_cache_enabled": settings.compression_cache_enabled,
        "auth_mode": settings.auth_mode,
        "api_key_injection": False,
        "api_key_headers_rejected": True,
        "api_keys_supported": False,
    }


@app.get("/stats")
async def stats() -> dict[str, Any]:
    snap = audit_logger.stats.snapshot()
    snap["result_cache"] = compressor.result_cache.stats()
    snap["preserve_anthropic_cache"] = settings.preserve_anthropic_cache
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


@app.post("/settings")
async def post_settings(request: Request) -> Response:
    body = await request.json()
    for k in ("input_compression", "output_compression", "jl_dedupe"):
        if k in body:
            _runtime[k] = bool(body[k])

    for engine_key, valid_levels in (
        ("caveman", _CAVEMAN_LEVELS),
        ("rtk", _RTK_LEVELS),
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

    _save_runtime()
    return JSONResponse(content=dict(_runtime))




@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    return HTMLResponse(content=_DASHBOARD_HTML)


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

    upstream_url = f"{settings.upstream}/{path}"
    if request.url.query:
        upstream_url = f"{upstream_url}?{request.url.query}"

    body_bytes = await request.body()
    started_perf = time.perf_counter()
    bytes_in = len(body_bytes) if body_bytes else 0
    request_audit = CompressionAudit(endpoint=path)
    outgoing_content: bytes | None = body_bytes if body_bytes else None

    try:
        if _runtime["input_compression"] and _should_transform_json_request(path, request.method, request.headers, body_bytes):
            payload = json.loads(body_bytes.decode("utf-8"))
            transformed, request_audit = compressor.compress_request_payload(
                payload,
                endpoint=path,
                jl_dedupe=_runtime["jl_dedupe"],
                caveman=_runtime["caveman"],
                rtk=_runtime["rtk"],
            )
            outgoing_content = json.dumps(transformed, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
            request_headers["content-type"] = "application/json"
    except Exception as exc:  # Keep the proxy useful even if compression fails.
        request_audit.events.clear()
        request_headers["x-middleout-warning"] = f"compression skipped: {type(exc).__name__}"
        outgoing_content = body_bytes if body_bytes else None

    method = request.method.upper()
    try:
        if _is_streaming_messages(path, outgoing_content):
            return await _streaming_forward(
                method=method,
                upstream_url=upstream_url,
                headers=request_headers,
                content=outgoing_content,
                request_audit=request_audit,
                path=path,
                started_perf=started_perf,
                bytes_in=bytes_in,
            )

        upstream_response = await app.state.http.request(
            method, upstream_url, headers=request_headers, content=outgoing_content
        )
        response_headers = _forward_response_headers(upstream_response.headers)
        response_headers.update(_compression_headers(request_audit, prefix="input"))

        response_content = upstream_response.content
        response_audit: CompressionAudit | None = None
        if _runtime["output_compression"] and _should_transform_json_response(path, upstream_response.headers, response_content):
            try:
                response_payload = upstream_response.json()
                transformed_response, response_audit = compressor.compress_response_payload(
                    response_payload, endpoint=path
                )
                if response_audit.touched:
                    response_content = json.dumps(
                        transformed_response, separators=(",", ":"), ensure_ascii=False
                    ).encode("utf-8")
                    response_headers.update(_compression_headers(response_audit, prefix="output"))
                    response_headers["content-type"] = "application/json"
            except Exception:
                response_audit = None

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
        )

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
) -> StreamingResponse:
    req = app.state.http.build_request(method, upstream_url, headers=headers, content=content)
    upstream_response = await app.state.http.send(req, stream=True)
    response_headers = _forward_response_headers(upstream_response.headers)
    response_headers.update(_compression_headers(request_audit, prefix="input"))

    bytes_out = 0

    async def body_iter():
        nonlocal bytes_out
        try:
            async for chunk in upstream_response.aiter_raw():
                bytes_out += len(chunk)
                yield chunk
        finally:
            audit_logger.record(
                method=method,
                path=path,
                status_code=upstream_response.status_code,
                request_audit=request_audit,
                request_id=upstream_response.headers.get("request-id"),
                latency_ms=(time.perf_counter() - started_perf) * 1000.0,
                bytes_in=bytes_in,
                bytes_out=bytes_out,
            )
            await upstream_response.aclose()

    return StreamingResponse(
        body_iter(),
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type=upstream_response.headers.get("content-type", "text/event-stream"),
    )


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
        # We may alter body bytes, so content-length is intentionally omitted above.
        forwarded[lower] = value
    return forwarded


def _compression_headers(audit: CompressionAudit, *, prefix: str) -> dict[str, str]:
    return {
        f"x-middleout-{prefix}-chars-saved": str(audit.chars_saved),
        f"x-middleout-{prefix}-events": str(len(audit.events)),
    }


def _should_transform_json_request(path: str, method: str, headers: Any, body: bytes) -> bool:
    if method.upper() != "POST" or not body:
        return False
    normalized = path.strip("/")
    if normalized not in {"v1/messages", "v1/messages/count_tokens"}:
        return False
    content_type = headers.get("content-type", "")
    return "application/json" in content_type.lower() or body[:1] in {b"{", b"["}


def _should_transform_json_response(path: str, headers: Any, body: bytes) -> bool:
    if not settings.output_compression_enabled or not body:
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
