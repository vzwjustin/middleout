"""Cache-key normalization for Anthropic Messages requests.

Goal: two request payloads that the upstream API would treat as semantically
identical must hash to the same cache key, even if the byte representations
differ. Two payloads that produce different upstream responses must hash to
different keys — never collide.

Strategy
--------
1. Drop client-supplied volatile fields. The only one in the Anthropic Messages
   schema is the top-level `metadata` object, which is opaque to the model and
   used for client-side tagging. Including it in the key would partition the
   cache by user_id / request_id without any change in upstream behavior.

2. Sort keys recursively so dict ordering doesn't matter.

3. Canonical JSON encoding: `separators=(",", ":")`, `sort_keys=True`,
   `ensure_ascii=False`.

4. sha256 the resulting bytes.

Fields we KEEP because they change the response:
- `model` (different model = different output)
- `system`, `messages`, `tools`, `tool_choice` (the prompt itself)
- `max_tokens`, `temperature`, `top_p`, `top_k`, `stop_sequences` (sampling)
- `stream` (response shape differs)
- `service_tier`, `anthropic_beta` (behavioral switches)
- Anything we don't explicitly drop (defensive — over-segmenting beats
  serving wrong results from cache).

We deliberately key on the post-compression payload (after middle-out, lingua,
etc.) rather than the original bytes. Two requests that compress identically
share a cache entry — that's the whole point. If the upstream would have given
the same response to both compressed forms, the cache hit is sound.

Cache validity
--------------
A cache hit replays a response upstream produced for an equivalent input. The
caller is responsible for any TTL policy (the L1 store carries `inserted_at`
on every record). For Anthropic, model versions are pinned in the request
payload itself, so the cache cannot accidentally serve an old model's output
when a request specifies a new model.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


# Top-level fields the cache key MUST NOT depend on. The Anthropic Messages
# schema is the source of truth; we only list fields known to be volatile or
# non-semantic.
_VOLATILE_TOP_LEVEL: frozenset[str] = frozenset({
    "metadata",
})


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `payload` with volatile fields stripped.

    The result is suitable for canonical-encoding into a cache key. The original
    payload is not modified.
    """
    if not isinstance(payload, dict):
        # Defensive: hash whatever was passed. Non-dict payloads aren't valid
        # Anthropic Messages requests; we let the upstream reject them rather
        # than silently dropping cache support.
        return {"__non_dict_payload__": True, "repr": repr(payload)}
    out: dict[str, Any] = {}
    for k, v in payload.items():
        if k in _VOLATILE_TOP_LEVEL:
            continue
        out[k] = v
    return out


def canonical_text(payload: dict[str, Any]) -> str:
    """Return the canonical JSON string used by both L1 and L2 cache layers.

    Exposed so the L2 layer can embed the *same* normalized text that L1 hashes,
    keeping the two layers' notion of "identical" aligned.
    """
    normalized = normalize_payload(payload)
    return json.dumps(
        normalized,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
        default=_json_default,
    )


def cache_key(payload: dict[str, Any]) -> str:
    """Return a SHA-256 hex digest of the canonical encoding of `payload`.

    Deterministic: same payload (modulo volatile fields) always produces the
    same key. Distinct payloads produce distinct keys with the cryptographic
    collision-resistance of SHA-256.
    """
    encoded = canonical_text(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_default(o: Any) -> Any:
    # Anthropic Messages payloads are pure JSON — strings, numbers, bools,
    # null, lists, dicts. If anything else (a bytes object, a datetime) slips
    # through, fall back to repr() so we never blow up the hash. The result
    # is still deterministic for that object.
    return repr(o)


__all__ = ["cache_key", "canonical_text", "normalize_payload"]
