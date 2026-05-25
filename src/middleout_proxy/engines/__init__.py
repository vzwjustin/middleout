"""Compression engine library.

Each module in this package exposes a stdlib-only, deterministic
``compress(text, *, level)`` callable that returns an :class:`EngineResult`.
The :data:`REGISTRY` maps an engine name to its callable so callers can
look engines up by string id.

The integration with the request/response compression pipeline happens
elsewhere; this package intentionally has no knowledge of ``Settings``,
caching, or HTTP semantics.
"""

from __future__ import annotations

from collections.abc import Callable

from . import (
    comment_strip,
    diff_compactor,
    json_collapse,
    log_collapse,
    path_collapse,
    stack_trace,
)
from .base import LEVELS, Engine, EngineResult

REGISTRY: dict[str, Callable[..., EngineResult]] = {
    "stack_trace": stack_trace.compress,
    "log_collapse": log_collapse.compress,
    "diff_compactor": diff_compactor.compress,
    "comment_strip": comment_strip.compress,
    "path_collapse": path_collapse.compress,
    "json_collapse": json_collapse.compress,
}

ENGINE_NAMES: tuple[str, ...] = tuple(REGISTRY.keys())


def apply_engine(name: str, text: str, level: str = "standard") -> EngineResult:
    """Run engine ``name`` against ``text`` at ``level``.

    Validates that ``name`` is a known engine and ``level`` is a known
    level. The ``off`` level is short-circuited here so engine modules can
    assume they only see active levels in their main body.
    """
    if name not in REGISTRY:
        raise ValueError(f"unknown engine: {name!r}")
    if level not in LEVELS:
        raise ValueError(f"level must be in {LEVELS}, got {level!r}")
    if level == "off":
        return EngineResult(
            text=text, original_chars=len(text), compressed_chars=len(text)
        )
    return REGISTRY[name](text, level=level)


__all__ = [
    "ENGINE_NAMES",
    "LEVELS",
    "REGISTRY",
    "Engine",
    "EngineResult",
    "apply_engine",
    "comment_strip",
    "diff_compactor",
    "json_collapse",
    "log_collapse",
    "path_collapse",
    "stack_trace",
]
