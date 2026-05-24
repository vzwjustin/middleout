# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A local Anthropic-compatible HTTP gateway (`middleout-proxy`, FastAPI) that Claude Code is pointed at via `ANTHROPIC_BASE_URL=http://127.0.0.1:8787`. It forwards to `https://api.anthropic.com/v1/messages`, doing two things on the way through:

1. **Lossy middle-out text compaction** of large user/tool-result blocks.
2. **JL-style sign-projection sketching** to find and replace near-duplicate large blocks within the same request.

The build is **strict subscription/OAuth passthrough only** — there is no API-key mode, and adding one is a non-goal. See "Auth invariants" below.

## Common commands

```bash
# install (editable, with dev deps)
pip install -e '.[dev]'

# run the proxy locally (must strip auth env vars — load_settings() refuses to start otherwise)
env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN -u PROXY_ANTHROPIC_API_KEY \
    -u PROXY_AUTH_MODE -u PROXY_FORCE_API_KEY \
    PROXY_UPSTREAM_BASE_URL=https://api.anthropic.com middleout-proxy

# tests
pytest                                          # all
pytest tests/test_compression.py                # one file
pytest tests/test_compression.py::test_middle_out_preserves_edges  # one test

# lint (ruff configured, line-length=100, target=py310)
ruff check src tests

# observability
curl http://127.0.0.1:8787/healthz
curl http://127.0.0.1:8787/stats
open http://127.0.0.1:8787/dashboard
```

`middleout-proxy` is the console-script entry point declared in `pyproject.toml` → `middleout_proxy.server:main`.

## Architecture

Request flow (`src/middleout_proxy/server.py`):

```
Claude Code → FastAPI catch-all /{path:path} → _forward_request_headers
  → (if POST /v1/messages or /v1/messages/count_tokens with JSON body)
       PayloadCompressor.compress_request_payload
         → middle_out_text  (compression.py)
         → RequestSketchIndex / signed_jl_projection  (jl.py)
         → optional caveman / RTK passes  (caveman.py, rtk.py)
  → httpx.AsyncClient.request OR build_request + stream=True for SSE
  → upstream Anthropic
  → (optional) compress_response_payload  [off by default; can break tools]
  → AuditLogger.record  (audit.py)  → .middleout-logs/audit.jsonl + ProxyStats
```

`/v1/messages` with `stream: true` is detected by `_is_streaming_messages` and forwarded through `_streaming_forward`, which uses `aiter_raw()` so SSE bytes are never buffered or rewritten. Everything else (e.g. `/v1/models`) is plain pass-through.

### Key modules

- `server.py` — FastAPI app, header forwarding, request/response routing, streaming forwarder, `/healthz`, `/stats`, `/settings` (GET+POST, persisted to `.middleout-logs/runtime_settings.json`), and `/dashboard`.
- `config.py` — `Settings` dataclass, `load_settings()`. **Runs `BLOCKED_AUTH_ENV_VARS` check at import-time of the server** — if any of `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `PROXY_ANTHROPIC_API_KEY`, `PROXY_AUTH_MODE`, `PROXY_FORCE_API_KEY` are set, the process refuses to start.
- `compression.py` — `PayloadCompressor`, `middle_out_text`, `CompressionAudit/Event`, `_CompressionResultCache` (bounded LRU keyed by sha256+params), and `_payload_cache_protection` (see "Anthropic prompt-cache invariant" below).
- `jl.py` — `tokenize`, `shingles`, `signed_jl_projection`, `cosine`, `RequestSketchIndex` (per-request, not global).
- `caveman.py` / `rtk.py` — optional deterministic text-prose compressors with discrete levels (`lite/standard/aggressive/ultra` and `minimal/standard/aggressive`). Both off by default.
- `audit.py` — `ProxyStats` snapshot for `/stats`; appends one JSONL line per request to `.middleout-logs/audit.jsonl` when `MIDDLEOUT_AUDIT=true`.
- `dashboard.py` — single inlined HTML string served at `/dashboard` for live toggles via `/settings`.

### Runtime-mutable settings

`_runtime` dict in `server.py` holds the booleans/levels users can flip live via `POST /settings` and the dashboard. It is seeded from `Settings` at startup and persisted to `.middleout-logs/runtime_settings.json`. Compression code paths read `_runtime[...]`, not `settings.*_enabled`, for these keys: `input_compression`, `output_compression`, `jl_dedupe`, `caveman`, `rtk`. When adding a new runtime-toggleable knob, mirror this pattern (validate in `post_settings`, persist via `_save_runtime`, thread into `compress_request_payload`).

## Auth invariants (do not weaken)

This is the entire reason the project exists. When editing `server.py` or `config.py`:

- `_forward_request_headers` **must** reject incoming `x-api-key` / `anthropic-api-key` headers with `StrictSubscriptionAuthError` and **must** require an `Authorization: Bearer ...` header. The error path returns 401 with `type: "strict_subscription_auth_error"`.
- `load_settings()` **must** raise on any `BLOCKED_AUTH_ENV_VARS`. Tests in `tests/test_strict_subscription_auth.py` parametrize over the full list.
- Never add code that reads an API key from env, injects an `x-api-key` header upstream, or falls back to API-key auth when OAuth is missing. The README's "What this strict build refuses" section is the contract.
- `/healthz` advertises `api_key_injection: false`, `api_key_headers_rejected: true`, `api_keys_supported: false`. These flags are part of the public contract — keep them accurate.

## Anthropic prompt-cache invariant

Anthropic prompt caching keys on the byte-identical prefix up through the last `cache_control` marker. `_payload_cache_protection` walks `system` blocks then `messages` in order and remembers the last marker; `_is_block_protected` then returns true for everything at-or-before it. **Compression skips protected blocks** so the upstream cache is not invalidated. When adding new compression passes (caveman, RTK, future engines) make sure they go through the same protection check — silently rewriting a protected block is a real bug, not a perf detail. Toggle: `MIDDLEOUT_PRESERVE_ANTHROPIC_CACHE` (default true).

## Conventions specific to this repo

- Python 3.10+, stdlib-only for `caveman.py`, `rtk.py`, `jl.py` (no numpy). Keep it that way — they're deliberately portable.
- Ruff line-length = 100, target = py310.
- Compression output **must be deterministic** — the LRU result cache (`_CompressionResultCache`) keys on input text + every parameter that influences output. Adding nondeterminism (timestamps, random seeds without a fixed seed) will silently produce wrong cache hits.
- Tests use a real `Settings()` constructed with overrides, not mocking. Follow that pattern — `test_compression.py` is the template.
- Audit logging logs **hashes and stats only** by default. `MIDDLEOUT_LOG_TEXT_SAMPLES=true` exists but should not be the default in any code path.
- `.middleout-logs/` is the on-disk side-effect directory (audit JSONL + persisted runtime settings). Don't move it without updating both `audit.py` and `server.py`.
