# MiddleOut Claude Proxy — strict Claude Code subscription build

A local Anthropic-compatible gateway for Claude Code that is intentionally **subscription/OAuth passthrough only**.

It sits between your terminal client and Anthropic's `/v1/messages` endpoint:

```text
Claude Code subscription login from /login
        │  ANTHROPIC_BASE_URL=http://127.0.0.1:8787
        ▼
MiddleOut Claude Proxy
        │  middle-out compaction + JL-style local dedupe
        │  forwards only Claude Code OAuth Authorization
        ▼
https://api.anthropic.com/v1/messages
```

## What this strict build refuses

This build has no API-key mode. It does not read, store, inject, or forward Anthropic API keys.

It fails fast when the proxy process has any of these auth variables set:

```bash
ANTHROPIC_API_KEY
ANTHROPIC_AUTH_TOKEN
PROXY_ANTHROPIC_API_KEY
PROXY_AUTH_MODE
PROXY_FORCE_API_KEY
```

It also rejects incoming `X-Api-Key` / `anthropic-api-key` headers from Claude Code. Requests must arrive with `Authorization: Bearer ...`, which is how OAuth credentials are passed through.

The proxy cannot cryptographically prove whether a Bearer token came from `/login` versus a manually set token, so you still need to verify Claude Code with `/status`. `/status` should show your Claude subscription/OAuth login, not custom API-key mode.

## Important idea check

TurboQuant is not an API-text compression trick. TurboQuant compresses model-side vectors, especially for vector quantization / attention math. The Johnson-Lindenstrauss idea is useful there because it preserves geometry of high-dimensional vectors after random projection.

A proxy to Anthropic cannot send random-projected vectors instead of prompt text because Anthropic's server expects normal Anthropic Messages JSON and has no decoder for your local sketch. So this project uses the JL idea where it *does* fit in a local proxy: fast near-duplicate detection and request-local redundancy removal. Actual prompt size reduction is done with lossy middle-out text compaction.

## What it does

- Exposes Anthropic-compatible endpoints:
  - `POST /v1/messages`
  - `POST /v1/messages/count_tokens`
  - pass-through for endpoints such as `GET /v1/models`
- Preserves Anthropic/Claude Code headers including `authorization`, `anthropic-version`, `anthropic-beta`, and Claude Code session headers.
- Forwards the incoming OAuth `Authorization` header unchanged.
- Rejects API-key headers instead of silently forwarding or injecting them.
- Middle-out compresses very large user/tool-result text blocks while preserving the beginning and end.
- Uses a Johnson-Lindenstrauss-style random sign projection sketch to detect near-duplicate large blocks already present in the same request and replace later copies with a short marker.
- Streams Claude responses through unchanged for `stream: true`.
- Logs only stats and hashes by default, not prompt text.

## Install

```bash
cd middleout-claude-proxy
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Claude Code subscription setup

First, make sure Claude Code is logged in normally:

```bash
claude
/login
/status
```

In `/status`, confirm it is using your Claude subscription/OAuth login.

### Terminal 1 — start the proxy with auth env stripped

```bash
cd middleout-claude-proxy

env \
  -u ANTHROPIC_API_KEY \
  -u ANTHROPIC_AUTH_TOKEN \
  -u PROXY_ANTHROPIC_API_KEY \
  -u PROXY_AUTH_MODE \
  -u PROXY_FORCE_API_KEY \
  PROXY_UPSTREAM_BASE_URL=https://api.anthropic.com \
  middleout-proxy
```

### Terminal 2 — start Claude Code through the proxy

```bash
env \
  -u ANTHROPIC_API_KEY \
  -u ANTHROPIC_AUTH_TOKEN \
  ANTHROPIC_BASE_URL=http://127.0.0.1:8787 \
  claude
```

Then run:

```text
/status
```

You want subscription/OAuth auth. If Claude Code reports API-key auth, clear `ANTHROPIC_API_KEY` from your shell, user settings, project settings, and any credential helper, then restart Claude Code.

## Health and stats

```bash
curl http://127.0.0.1:8787/healthz
curl http://127.0.0.1:8787/stats
```

`/healthz` includes:

```json
{
  "auth_mode": "subscription_oauth_passthrough_only",
  "api_key_injection": false,
  "api_key_headers_rejected": true,
  "api_keys_supported": false
}
```

## Config

Copy `.env.example` into your shell or export only what you need.

Key settings:

```bash
MIDDLEOUT_INPUT_COMPRESSION=true
MIDDLEOUT_MAX_TEXT_CHARS=12000
MIDDLEOUT_COMPRESS_SYSTEM=false
MIDDLEOUT_COMPRESS_TOOL_RESULTS=true

MIDDLEOUT_JL_DEDUPE=true
MIDDLEOUT_JL_DIMS=512
MIDDLEOUT_JL_SIMILARITY=0.985
MIDDLEOUT_JL_MIN_CHARS=4000

# Off by default because it can confuse tools/agents that expect exact output.
MIDDLEOUT_OUTPUT_COMPRESSION=false
```

## How the compression works

### Middle-out text compaction

For a text block over `MIDDLEOUT_MAX_TEXT_CHARS`, the proxy keeps the beginning and end and replaces the middle with a marker like:

```text
[... middle-out compressed locally: omitted 18420 chars; original_chars=30420; sha256=...; not reversible by the model ...]
```

This is useful for logs, stack traces, diffs, generated files, and tool outputs where the start/end often carry enough signal. It is lossy. The model cannot recover omitted text.

### JL-style local sketching

The proxy tokenizes large text blocks into word shingles, hashes them into a fixed-dimensional signed random projection, normalizes that vector, then compares cosine similarity against earlier large blocks in the same request.

When a later block is almost identical to an earlier block, it can be replaced with:

```text
[Near-duplicate content omitted locally by JL-style request sketch. Similar to earlier block at ...]
```

This borrows the useful part of JL for a proxy: preserving similarity under cheap random projection. It does not send compressed vectors to Claude.

## Run tests

```bash
pip install -e '.[dev]'
pytest
```

## Brain proxy — Phase 2/3/4 (cache, providers, cost)

Phase 1 (cache-aware compression + transparent passthrough) is the legacy core
above. The rest of the codebase scaffolds the "all-in-1 brain" surface area
described in `CLAUDE.md`:

- **Cache wall enforcement** (`cache_wall.py`, `volatile.py`) — parses cache
  control breakpoints, refuses to mutate anything left of the wall. Auto-insert
  is opt-in via `BRAIN_AUTO_INSERT_WALL`.
- **LLMLingua-2** (`lingua.py`) — opt-in compression of the volatile tail.
  Requires `pip install -e '.[lingua]'` (downloads a ~200MB BERT model on
  first use). Configure with `BRAIN_LINGUA_ENABLED`, `BRAIN_LINGUA_RATIO`.
- **L1 exact-match response cache** (`cache/l1.py`) — SQLite-backed,
  bounded entries with LRU eviction. Key = SHA-256 of normalized post-compression
  payload. Off by default; enable with `BRAIN_L1_CACHE_ENABLED=true`.
- **L2 semantic response cache** (`cache/l2.py`) — embedding-based ANN lookup.
  Backends: in-memory (default) or Qdrant (`pip install -e '.[qdrant]'`).
  Embedders: `hash` (deterministic stdlib pseudo-embedder, default) or
  `openai` (`pip install -e '.[openai]'`, needs `OPENAI_API_KEY`).
- **Provider adapters** (`providers/`) — Anthropic identity adapter ships;
  OpenAI, Gemini, Ollama are scaffolded and respond with `501 adapter_not_implemented`
  when targeted via `X-Brain-Model-Hint`. The proxy's IR *is* the Anthropic
  Messages schema.
- **Cost tracker** (`cost.py`) — bakes in per-million-token prices for
  Anthropic/OpenAI/Gemini models. Stamps `x-brain-cost-usd` on every successful
  `/v1/messages` response and exposes `/cost`, `/cost/reset`, `/budget`.
- **Rate-limit & budget hooks** (`rate_limit.py`, `budget.py`) — operator can
  set char/token caps that flag exceeded requests in `/budget`.

### Brain endpoints

| Path                | Description                                          |
| ------------------- | ---------------------------------------------------- |
| `GET  /healthz`     | Lifespan + phase + provider list + cache wiring      |
| `GET  /stats`       | Audit + cache + compression counters                 |
| `GET  /metrics`     | Prometheus-format snapshot                           |
| `POST /preview`     | Dry-run compression sizing (never touches the wire)  |
| `GET  /cost`        | Cumulative spend by model + budget snapshot          |
| `POST /cost/reset`  | Zero the cost counters                               |
| `GET  /providers`   | Registered adapters + routing rules                  |
| `GET  /cache/stats` | L1 + L2 cache state                                  |
| `POST /cache/purge` | Drop every L1 entry                                  |
| `GET  /budget`      | Process-level budget snapshot                        |
| `GET  /dashboard`   | Live HTML dashboard (engines, cost, cache, traffic)  |

### TOML config (Phase 1 spec)

Settings can live in a `middleout.toml` file. Lookup order:

1. `MIDDLEOUT_CONFIG` env var (explicit path)
2. `./middleout.toml`
3. `~/.config/middleout/middleout.toml`

Env vars still override the TOML defaults. See `config.py:_TOML_FIELD_MAP` for
the full schema; example:

```toml
[server]
host = "127.0.0.1"
port = 8787

[lingua]
enabled = true
ratio = 0.5

[l1_cache]
enabled = true
db_path = ".middleout-logs/l1.sqlite"

[l2_cache]
enabled = true
similarity_threshold = 0.97
```

### Provider routing

A client can pin a request to a specific adapter via header:

```bash
curl -H 'X-Brain-Model-Hint: openai/gpt-4o' ...
```

Adapter selection is exposed via `GET /providers`; only the Anthropic adapter
currently round-trips end-to-end. Non-Anthropic hints respond `501`.

## Files

```text
src/middleout_proxy/
  server.py            FastAPI gateway, lifespan, streaming pass-through, all routes
  compression.py       request/response compaction (middle-out + JL + engines)
  jl.py                JL-style sign projection sketches
  cache_wall.py        Anthropic cache_control parser (the cache wall)
  volatile.py          compresses the volatile tail right of the wall
  lingua.py            LLMLingua-2 wrapper (opt-in)
  config.py            TOML + env + defaults; strict auth guardrails
  audit.py             JSONL audit + rolling timeseries
  dashboard.py         single-page HTML dashboard
  preview.py           /preview endpoint helpers
  metrics.py           Prometheus-format renderer
  policies.py          per-model/per-endpoint policy router
  adaptive.py          context-pressure level decider
  rate_limit.py        backoff + queue helpers
  budget.py            char/token cap tracker
  cost.py              price table + CostTracker
  preview.py           dry-run sizing
  providers/           adapter package (anthropic/openai/google/ollama)
  cache/               L1 (SQLite), L2 (in-memory or Qdrant), normalization
  engines/             secondary compression engines (caveman, rtk, ...)
  sim/                 MinHash / SimHash similarity primitives
tests/                 680+ pytest cases, runs in <3s
benchmarks/            offline compression-ratio benchmarks
docs/                  design notes
```

## Caveats

- This is an MVP, not a drop-in replacement for model-side KV-cache quantization.
- Do not compress system prompts unless you are intentionally experimenting.
- Output compression is disabled by default because it can break tool calling or agent loops.
- Prompt compaction can change model behavior. Start with high thresholds and inspect `.middleout-logs/audit.jsonl`.
- The proxy can reject API-key headers, but only Claude Code's `/status` can tell you which credential source Claude Code selected before it sent the request.
