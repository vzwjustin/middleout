"""Process-level cumulative usage budget.

Tracks total characters and tokens consumed by the current process. The
integration layer is expected to call :meth:`UsageBudget.record` after each
upstream completion (or after each compression pass — pick one and stick with
it).

Thread-safe via a single :class:`threading.Lock`. ``None`` for either limit
means "no limit on this axis"; the corresponding entry in :meth:`remaining`
returns ``None`` as well.
"""

from __future__ import annotations

import threading
from typing import Any


class UsageBudget:
    """Cumulative char/token counter with optional per-axis ceilings."""

    def __init__(
        self,
        *,
        char_limit: int | None = None,
        token_limit: int | None = None,
    ) -> None:
        if char_limit is not None and char_limit < 0:
            raise ValueError("char_limit must be non-negative or None")
        if token_limit is not None and token_limit < 0:
            raise ValueError("token_limit must be non-negative or None")
        self.char_limit: int | None = char_limit
        self.token_limit: int | None = token_limit
        self._chars: int = 0
        self._tokens: int = 0
        self._lock = threading.Lock()

    def record(self, *, chars: int, tokens: int) -> None:
        """Add ``chars`` and ``tokens`` to the running totals. Both must be non-negative."""
        if chars < 0 or tokens < 0:
            raise ValueError("chars and tokens must be non-negative")
        with self._lock:
            self._chars += int(chars)
            self._tokens += int(tokens)

    def remaining(self) -> dict[str, int | None]:
        """Return ``{"chars": ..., "tokens": ...}``. ``None`` axes carry through as ``None``."""
        with self._lock:
            chars_remaining = (
                None if self.char_limit is None else max(0, self.char_limit - self._chars)
            )
            tokens_remaining = (
                None if self.token_limit is None else max(0, self.token_limit - self._tokens)
            )
        return {"chars": chars_remaining, "tokens": tokens_remaining}

    def exceeded(self) -> bool:
        """``True`` when usage has reached or passed any configured limit."""
        with self._lock:
            return self._exceeded_locked()

    def reset(self) -> None:
        """Zero the counters. Limits stay where they were."""
        with self._lock:
            self._chars = 0
            self._tokens = 0

    def snapshot(self) -> dict[str, Any]:
        """A point-in-time view of the budget. Safe to log."""
        with self._lock:
            return {
                "chars_used": self._chars,
                "tokens_used": self._tokens,
                "char_limit": self.char_limit,
                "token_limit": self.token_limit,
                "exceeded": self._exceeded_locked(),
            }

    # -- internals ---------------------------------------------------------

    def _exceeded_locked(self) -> bool:
        if self.char_limit is not None and self._chars >= self.char_limit:
            return True
        if self.token_limit is not None and self._tokens >= self.token_limit:
            return True
        return False


__all__ = ["UsageBudget"]
