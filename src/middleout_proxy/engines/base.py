"""Engine protocol and shared result type for the compression engine library.

Each engine in this package exposes a ``compress(text, *, level)`` callable that
returns an :class:`EngineResult`. Engines are stdlib-only and deterministic so
they can sit downstream of the LRU cache in ``compression.py`` without breaking
its keying contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

LEVELS = ("off", "lite", "standard", "aggressive")


@dataclass(frozen=True)
class EngineResult:
    """Result of running a single engine on a single block of text.

    ``original_chars`` and ``compressed_chars`` are reported in characters (not
    bytes) so they match how the rest of the proxy accounts savings.
    """

    text: str
    note: str = ""
    original_chars: int = 0
    compressed_chars: int = 0

    @property
    def chars_saved(self) -> int:
        return max(0, self.original_chars - self.compressed_chars)

    @property
    def changed(self) -> bool:
        return self.text != "" and self.original_chars != self.compressed_chars


class Engine(Protocol):
    """Callable protocol for an engine module's top-level ``compress`` fn."""

    name: str

    def __call__(self, text: str, *, level: str = "standard") -> EngineResult: ...


def make_result(original: str, compressed: str, note: str = "") -> EngineResult:
    """Convenience constructor that fills in char counts from the two strings."""
    return EngineResult(
        text=compressed,
        note=note,
        original_chars=len(original),
        compressed_chars=len(compressed),
    )


def validate_level(level: str) -> None:
    if level not in LEVELS:
        raise ValueError(f"level must be in {LEVELS}, got {level!r}")
