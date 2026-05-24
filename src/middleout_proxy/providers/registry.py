"""Adapter registry + routing selector.

Each adapter module calls :func:`register` at import time. The integration
layer calls :func:`select_adapter` once per request with the model hint and
the body model; it returns the resolved adapter (never raises — falls back to
the default Anthropic adapter when no rule matches).
"""

from __future__ import annotations

import fnmatch
from typing import Any

from .base import Adapter

REGISTRY: dict[str, Adapter] = {}

# Ordered list of (glob_pattern, provider_name). First match wins. Filled by
# :func:`register_route` so model strings like ``gpt-4o`` route to OpenAI.
_ROUTES: list[tuple[str, str]] = []


def register(adapter: Adapter, *, model_globs: tuple[str, ...] = ()) -> None:
    """Register `adapter` under its `name`.

    Optional `model_globs` register routing patterns — e.g. ``("gpt-*",
    "openai/*")`` routes any model whose id matches one of those globs to this
    adapter. The Anthropic adapter is registered last (or has the broadest
    fallback pattern) so unknown models still flow through it.
    """
    REGISTRY[adapter.name] = adapter
    for glob in model_globs:
        _ROUTES.append((glob, adapter.name))


def select_adapter(*, model: str, model_hint: str | None = None) -> Adapter:
    """Pick an adapter for a request.

    Resolution order:

    1. ``model_hint`` exactly matches an adapter name (e.g. ``X-Brain-Model-Hint:
       openai`` — explicit operator override).
    2. ``model_hint`` matches a registered model glob (e.g. ``X-Brain-Model-
       Hint: gpt-4o-mini`` — explicit model id override that should be routed
       like the body model would be).
    3. ``model`` (from the body) matches a registered model glob.
    4. Default adapter: Anthropic.
    """
    if model_hint:
        # Direct adapter name match takes priority.
        if model_hint in REGISTRY:
            return REGISTRY[model_hint]
        for glob, name in _ROUTES:
            if fnmatch.fnmatch(model_hint, glob) and name in REGISTRY:
                return REGISTRY[name]

    if model:
        for glob, name in _ROUTES:
            if fnmatch.fnmatch(model, glob) and name in REGISTRY:
                return REGISTRY[name]

    # Default fallback. The Anthropic adapter must always be present.
    if "anthropic" in REGISTRY:
        return REGISTRY["anthropic"]
    raise RuntimeError(
        "No anthropic adapter registered. The providers package must always "
        "register at least the Anthropic identity adapter."
    )


def routes_snapshot() -> dict[str, Any]:
    """Diagnostic snapshot of routing state for `/healthz` and tests."""
    return {
        "adapters": sorted(REGISTRY.keys()),
        "routes": [{"glob": g, "adapter": n} for g, n in _ROUTES],
    }


def clear() -> None:
    """Reset the registry. Tests only."""
    REGISTRY.clear()
    _ROUTES.clear()


__all__ = ["REGISTRY", "clear", "register", "routes_snapshot", "select_adapter"]
