"""In-process token-bucket rate limiter.

Local DOS protection only. Not a distributed rate limiter; not a fairness
guarantee — it just keeps a single process from drowning in cheap clients.

The integration layer is expected to identify each client by a *hashed*
authorization token::

    client_key = hashlib.sha256(authorization_value.encode()).hexdigest()[:16]

This module **must never receive a raw auth token**. The keying is the
caller's responsibility precisely to keep secrets out of this code path.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Any


class TokenBucket:
    """Classic token bucket. Async-safe via :class:`asyncio.Lock`."""

    def __init__(self, *, capacity: int, refill_per_second: float) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_per_second <= 0:
            raise ValueError("refill_per_second must be positive")
        self.capacity: float = float(capacity)
        self.refill_per_second: float = float(refill_per_second)
        self._tokens: float = float(capacity)
        self._last: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, cost: int = 1) -> bool:
        """Try to take ``cost`` tokens. Returns ``True`` on success, ``False`` if
        the bucket is empty. Never waits."""
        if cost <= 0:
            return True
        async with self._lock:
            self._refill_locked()
            if self._tokens >= cost:
                self._tokens -= cost
                return True
            return False

    def reset(self) -> None:
        """Refill to capacity and reset the refill clock. Sync — safe to call any time."""
        self._tokens = self.capacity
        self._last = time.monotonic()

    @property
    def available(self) -> float:
        """A best-effort, lock-free snapshot of the available token count."""
        now = time.monotonic()
        elapsed = max(0.0, now - self._last)
        return min(self.capacity, self._tokens + elapsed * self.refill_per_second)

    # -- internals ---------------------------------------------------------

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_second)
            self._last = now


class RequestLimiter:
    """Per-client (pre-hashed authorization) token-bucket limiter.

    Buckets are created lazily on first ``check`` for a given key. There is no
    automatic eviction; the integration layer can wrap this with a janitor if
    long-running processes accumulate too many distinct clients.
    """

    def __init__(self, *, capacity: int = 60, refill_per_second: float = 1.0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_per_second <= 0:
            raise ValueError("refill_per_second must be positive")
        self.capacity = int(capacity)
        self.refill_per_second = float(refill_per_second)
        self._buckets: OrderedDict[str, TokenBucket] = OrderedDict()
        self._created_at: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def check(self, client_key: str) -> bool:
        """Return ``True`` if the client is within budget, ``False`` if exhausted.

        ``client_key`` must already be a hashed/truncated identifier — never a
        raw bearer token.
        """
        if not isinstance(client_key, str) or not client_key:
            raise ValueError("client_key must be a non-empty string")

        bucket = self._buckets.get(client_key)
        if bucket is None:
            async with self._lock:
                bucket = self._buckets.get(client_key)
                if bucket is None:
                    bucket = TokenBucket(
                        capacity=self.capacity,
                        refill_per_second=self.refill_per_second,
                    )
                    self._buckets[client_key] = bucket
                    self._created_at[client_key] = time.monotonic()
                self._buckets.move_to_end(client_key)
        else:
            # Cheap LRU touch without taking the outer lock.
            try:
                self._buckets.move_to_end(client_key)
            except KeyError:
                pass
        return await bucket.acquire(1)

    def stats(self) -> dict[str, Any]:
        """Summary of bucket bookkeeping. Not a hot-path metric."""
        oldest = min(self._created_at.values(), default=None)
        return {
            "active_buckets": len(self._buckets),
            "oldest_created_at": oldest,
            "capacity": self.capacity,
            "refill_per_second": self.refill_per_second,
        }


__all__ = ["TokenBucket", "RequestLimiter"]
