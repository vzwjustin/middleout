"""L1 exact-match response cache backed by SQLite.

Stores responses keyed by the SHA-256 of a normalized request payload (see
`normalize.cache_key`). Bounded by total entries; oldest-access-first eviction.

Concurrency
-----------
SQLite in WAL mode handles concurrent reads cleanly. Writes are serialized by
the SQLite library. We don't take a Python-level lock — the per-connection
`Connection` is created fresh per call when needed, and `check_same_thread=False`
is set so the AsyncClient event loop can reuse the same connection if desired.

Streaming responses
-------------------
This phase does NOT cache streaming responses. The SSE chunk boundary
preservation invariant from the user spec ("byte-identical to what the API
returned, including streaming chunk boundaries when possible") deserves its
own dedicated implementation — recording chunks and timestamps and replaying
through `StreamingResponse` — and is out of scope for the first L1 cut.

Caller contract
---------------
- `put(key, response)` is best-effort: storage failures are swallowed and the
  request is unaffected.
- `get(key)` returns `None` for miss or any storage error.
- Responses are stored as bytes; the caller is responsible for serialization
  of headers (the `CachedResponse.headers` dict is JSON-encoded internally).
- Never store anything containing raw auth headers — strip them before
  calling `put` (the integration layer does this).
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS l1_cache (
    cache_key TEXT PRIMARY KEY,
    status_code INTEGER NOT NULL,
    headers_json TEXT NOT NULL,
    body BLOB NOT NULL,
    media_type TEXT,
    inserted_at REAL NOT NULL,
    last_hit_at REAL NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0,
    body_bytes INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS l1_cache_last_hit ON l1_cache(last_hit_at);
"""


# Headers we refuse to cache even if the caller passes them in. Defense in
# depth: if upstream ever returns an auth header, it must not survive a cache
# round-trip.
_NEVER_CACHE_HEADERS: frozenset[str] = frozenset({
    "authorization",
    "x-api-key",
    "anthropic-api-key",
    "proxy-authorization",
    "set-cookie",
})


@dataclass
class CachedResponse:
    """One cached upstream response.

    `headers` are stored exactly as returned by upstream (except auth headers,
    which are dropped). `body` is the response body bytes (already decompressed
    if upstream sent gzip — the proxy strips content-encoding on the way out).
    """

    status_code: int
    headers: dict[str, str]
    body: bytes
    media_type: str | None = None
    inserted_at: float = field(default_factory=time.time)
    last_hit_at: float = field(default_factory=time.time)
    hit_count: int = 0


class L1Cache:
    """SQLite-backed L1 response cache.

    `db_path=":memory:"` runs entirely in memory — handy for tests and for
    "cache-on, persistence-off" deployments. File-backed paths persist across
    proxy restarts.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        max_entries: int = 10_000,
        max_body_bytes: int = 5 * 1024 * 1024,  # refuse to cache >5MB responses
    ) -> None:
        self.db_path = str(db_path)
        self.max_entries = int(max_entries)
        self.max_body_bytes = int(max_body_bytes)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        for stmt in _SCHEMA.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt)

    # -- public API -------------------------------------------------------

    def get(self, key: str) -> CachedResponse | None:
        """Return the cached response for `key`, or None on miss/error."""
        try:
            cur = self._conn.execute(
                "SELECT status_code, headers_json, body, media_type, inserted_at, "
                "last_hit_at, hit_count FROM l1_cache WHERE cache_key = ?",
                (key,),
            )
            row = cur.fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        status_code, headers_json, body, media_type, inserted_at, _, hit_count = row
        try:
            headers = json.loads(headers_json)
        except json.JSONDecodeError:
            return None
        now = time.time()
        try:
            self._conn.execute(
                "UPDATE l1_cache SET last_hit_at = ?, hit_count = hit_count + 1 "
                "WHERE cache_key = ?",
                (now, key),
            )
        except sqlite3.Error:
            pass  # don't fail the read on a stat-update failure
        return CachedResponse(
            status_code=int(status_code),
            headers=headers,
            body=bytes(body),
            media_type=media_type,
            inserted_at=float(inserted_at),
            last_hit_at=now,
            hit_count=int(hit_count) + 1,
        )

    def put(self, key: str, response: CachedResponse) -> None:
        """Store `response` under `key`. Silently no-op on storage errors.

        Refuses to cache:
        - 5xx responses (transient upstream errors shouldn't be replayed)
        - 4xx responses (auth failures, validation errors — could leak)
        - Bodies larger than `max_body_bytes`
        - Headers containing auth-leaking entries (defensive — stripped first)
        """
        if not 200 <= response.status_code < 300:
            return
        if len(response.body) > self.max_body_bytes:
            return

        clean_headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower() not in _NEVER_CACHE_HEADERS
        }
        headers_json = json.dumps(clean_headers, ensure_ascii=False, sort_keys=True)
        now = time.time()
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO l1_cache "
                "(cache_key, status_code, headers_json, body, media_type, "
                " inserted_at, last_hit_at, hit_count, body_bytes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    int(response.status_code),
                    headers_json,
                    bytes(response.body),
                    response.media_type,
                    now,
                    now,
                    0,
                    len(response.body),
                ),
            )
            self._evict_if_over()
        except sqlite3.Error:
            return

    def stats(self) -> dict[str, Any]:
        try:
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(body_bytes), 0), "
                "COALESCE(SUM(hit_count), 0) FROM l1_cache"
            ).fetchone()
        except sqlite3.Error:
            return {"entries": 0, "body_bytes": 0, "total_hits": 0}
        return {
            "entries": int(row[0]),
            "body_bytes": int(row[1]),
            "total_hits": int(row[2]),
            "max_entries": self.max_entries,
            "max_body_bytes": self.max_body_bytes,
        }

    def clear(self) -> None:
        try:
            self._conn.execute("DELETE FROM l1_cache")
        except sqlite3.Error:
            pass

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    # -- internals --------------------------------------------------------

    def _evict_if_over(self) -> None:
        """LRU-by-last_hit eviction down to `max_entries`.

        Cheap when we're under the cap (one COUNT query). Expensive only when
        we just crossed the cap (one DELETE with a subquery).
        """
        try:
            (count,) = self._conn.execute("SELECT COUNT(*) FROM l1_cache").fetchone()
        except sqlite3.Error:
            return
        excess = int(count) - self.max_entries
        if excess <= 0:
            return
        try:
            self._conn.execute(
                "DELETE FROM l1_cache WHERE cache_key IN ("
                "  SELECT cache_key FROM l1_cache ORDER BY last_hit_at ASC LIMIT ?"
                ")",
                (excess,),
            )
        except sqlite3.Error:
            pass


__all__ = ["CachedResponse", "L1Cache"]
