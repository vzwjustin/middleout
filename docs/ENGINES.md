# New Compression Engines

This doc covers the three engines added under `army/engines`:

1. `json_aware` — JSON minification + whitespace collapse.
2. `lsh_dedupe` — banded-LSH minhash near-duplicate detection across blocks.
3. `adaptive` — pure-function policy that decides which engines/levels to run
   based on payload size and model context pressure.

They are NOT wired into `server.py`/`compression.py` yet. This document is the
integration spec for a follow-up central-wiring change.

All three engines satisfy the existing project invariants:

- **Cache invariant.** No engine touches a block; callers must consult
  `compression._is_block_protected` and skip protected blocks. `lsh_dedupe`'s
  `dedupe_blocks` exposes a `protected: set[int]` arg for this purpose.
- **Determinism.** All engines are pure / stateless / use only stdlib hashing
  (`blake2b`). Same input gives same output, so the existing
  `_CompressionResultCache` remains correct.
- **No I/O, no env, no time, no randomness.** `adaptive` is documented as a
  pure function and `_pressure` reads only the payload.

---

## Engine 1: `json_aware`

**Module:** `src/middleout_proxy/json_aware.py`

### Public API

```python
def compress(text: str, level: str) -> tuple[str, dict]:
    """Returns (new_text, stats={chars_in, chars_out, blocks_found})."""
```

`level` must be one of `safe`, `standard`, `aggressive` — `ValueError` on
unknown level (matches caveman/rtk).

### Levels

| Level        | Behavior                                                                 |
| ------------ | ------------------------------------------------------------------------ |
| `safe`       | JSON minify only when the block parses. No whitespace touch.             |
| `standard`   | + collapse blank lines / trim trailing whitespace in prose and in fenced blocks whose language is NOT in `_WHITESPACE_SAFE_LANGS` (`python`, `py`, `yaml`, `yml`, `makefile`, `make`, `haml`, `coffee`/`coffeescript`, `fsharp`/`f#`, `sass`). |
| `aggressive` | + strip `//` and `/* */` JSONC comments and trailing commas before re-parse. Refuses when a comment marker appears inside a string literal. |

### Behavior

- Splits on triple-backtick fences via `_FENCE_SPLIT_RE` (the same
  `re.compile(r"```")` pattern used by `caveman`/`rtk`).
- Tries to minify each fenced block whose lang is `json`/`jsonc`/`json5` OR
  whose body parses as JSON. Aggressive level also tries when the body starts
  with `{`/`[`.
- Partial/truncated JSON is left unchanged (handled via `_parse_json_strict`
  which catches `ValueError`/`RecursionError`).

### Integration hook point

In `src/middleout_proxy/compression.py`, add a new pass to
`_compress_text_with_dedupe` between `_compress_text` and `_apply_caveman`:

- `compression.py:455` (right after the `_compress_text(...)` call inside
  `_compress_text_with_dedupe`).
- Mirror the existing `_apply_caveman` / `_apply_rtk` helpers
  (`compression.py:484` and `compression.py:508`) — same shape: read
  `_json_aware_active` (set in `compress_request_payload`), call
  `compress(...)`, emit a `CompressionEvent(mode="json-aware", ...)`.
- Extend `_build_cache_key` (`compression.py:461`) to include the new
  `enabled` + `level` so cached outputs don't leak across config changes.

---

## Engine 2: `lsh_dedupe`

**Module:** `src/middleout_proxy/lsh_dedupe.py`

### Public API

```python
class LSHDedupeIndex:
    def __init__(self, level: str = "standard") -> None: ...
    def add(self, block_id, text: str) -> None: ...
    def find_near_duplicate(self, text: str) -> Optional[tuple[block_id, similarity]]: ...

def dedupe_blocks(
    blocks: list[dict],
    level: str = "standard",
    protected: set[int] | None = None,
) -> tuple[list[dict], dict]:
    """Replace later near-dup blocks with a marker. Returns (new_blocks, stats)."""
```

`ValueError` on unknown level (both class init and `dedupe_blocks`).

### Levels

Banded-LSH parameters. Signature width fixed at 128.

| Level          | Threshold | Bands x Rows |
| -------------- | --------- | ------------ |
| `conservative` | 0.95      | 8 x 16       |
| `standard`     | 0.88      | 16 x 8       |
| `aggressive`   | 0.80      | 32 x 4       |

### Behavior

- Minhash: 128 independent `blake2b` hashes per block, parameterized by a
  per-hash `person=` bytes value so each gives an independent permutation.
- Shingle width 5 via the existing `jl.tokenize` + `jl.shingles`.
- Candidate retrieval: any band whose hashed `rows`-tuple matches.
- Verification: jaccard estimate via `same_positions / signature_width`; only
  candidates meeting `threshold` are returned.
- `dedupe_blocks` keeps the FIRST occurrence intact and replaces later
  near-dups with `[duplicate of earlier block at <id>, ~N chars, similarity 0.XX]`.
- `protected` indices are added to the index (so later blocks can match them)
  but are themselves never replaced.

### Integration hook point

In `src/middleout_proxy/compression.py:295` (the `messages` loop in
`compress_request_payload`):

- Today's per-block JL sketch lives inside `_compress_text_with_dedupe`. The
  new LSH pass operates at the WHOLE-REQUEST level across content blocks. Add
  a new pre-pass before the messages loop:
  1. Collect every textual block (system items + per-message content blocks)
     into a single flat `list[dict]` with stable indices.
  2. Compute the `protected` set by walking the same indices through
     `_is_block_protected` (`compression.py:150`).
  3. Call `lsh_dedupe.dedupe_blocks(blocks, level=runtime["lsh_level"],
     protected=protected)`.
  4. Write the deduped blocks back into the payload (or use the returned
     `new_blocks` as the working copy).
- LSH dedupe should run AFTER `json_aware` minify (so identical
  pretty-printed-but-different-spacing JSON blocks collide) and BEFORE
  caveman/rtk (so dedupe markers aren't themselves abbreviated).

---

## Engine 3: `adaptive`

**Module:** `src/middleout_proxy/adaptive.py`

### Public API

```python
def should_compress(payload: dict) -> bool:
    """False when total payload text < 2KB."""

def decide_levels(payload: dict) -> dict:
    """Returns {middle_out, caveman, rtk, json_aware, lsh, jl_dedupe}."""
```

Both are pure functions. No I/O, no time, no randomness, no env reads.

### Context pressure to tier table

`pressure = (total_chars / 4) / context_window` (4 chars/token approximation).

| Pressure   | Tier         | middle_out  | caveman     | rtk         | json_aware  | lsh           | jl_dedupe |
| ---------- | ------------ | ----------- | ----------- | ----------- | ----------- | ------------- | --------- |
| `< 0.40`   | `lenient`    | `off`       | `lite`      | `minimal`   | `safe`      | `conservative`| False     |
| `< 0.60`   | `standard`   | `safe`      | `standard`  | `standard`  | `standard`  | `standard`    | True      |
| `< 0.80`   | `aggressive` | `safe`      | `aggressive`| `aggressive`| `aggressive`| `aggressive`  | True      |
| `>= 0.80`  | `max`        | `aggressive`| `ultra`     | `aggressive`| `aggressive`| `aggressive`  | True      |

### Model context windows (prefix match)

All currently supported families default to 200,000 tokens. Unknown models
also default to 200,000. The prefix table covers:

- `claude-3-5-sonnet`, `claude-3-7-sonnet`, `claude-3-7`,
- `claude-3-opus`, `claude-3-haiku`, `claude-3`,
- `claude-opus-4*`, `claude-sonnet-4*`, `claude-haiku-4*`.

### Integration hook point

In `src/middleout_proxy/server.py`, inside the POST handler for
`/v1/messages` and `/v1/messages/count_tokens`, BEFORE the call to
`compressor.compress_request_payload`:

```python
from middleout_proxy.adaptive import decide_levels, should_compress

if settings.adaptive_enabled and not should_compress(payload):
    transformed, audit = payload, CompressionAudit(endpoint=endpoint)
else:
    if settings.adaptive_enabled:
        chosen = decide_levels(payload)
        runtime_overrides = {
            "jl_dedupe": chosen["jl_dedupe"],
            "caveman": {"enabled": chosen["caveman"] != "lite", "level": chosen["caveman"]},
            "rtk":     {"enabled": chosen["rtk"]     != "minimal", "level": chosen["rtk"]},
            # plus json_aware + lsh once those runtime keys are added
        }
        transformed, audit = compressor.compress_request_payload(
            payload, endpoint=endpoint, **runtime_overrides
        )
    else:
        transformed, audit = compressor.compress_request_payload(payload, endpoint=endpoint)
```

The exact integration depends on what new runtime keys are added to the
`_runtime` dict in `server.py` (today's: `input_compression`,
`output_compression`, `jl_dedupe`, `caveman`, `rtk` — see CLAUDE.md
"Runtime-mutable settings").

---

## Suggested wiring order

1. Add `json_aware` config keys to `Settings` + `_runtime` (`server.py`),
   wire the engine into `_compress_text_with_dedupe` after `_compress_text`.
2. Add `lsh` config keys to `Settings` + `_runtime`, wire the engine as a
   pre-pass over flattened content blocks in `compress_request_payload`.
3. Add `adaptive` config key (single bool) to `Settings` + `_runtime`. In the
   POST handler, gate the compressor call on `should_compress` and pass
   per-engine levels from `decide_levels`.

Each step is testable on its own. Don't merge step 3 until 1 and 2 are
landed; otherwise `decide_levels` returns levels for engines that aren't
wired in.
