"""Provider adapter scaffolding (Phase 3).

The internal IR is the Anthropic Messages schema. Adapters convert outgoing
requests to a provider's native format, and incoming responses back to the IR
so downstream code (cache, audit, dashboard) keeps speaking one shape.

This package only contains scaffolding for now — the Anthropic "adapter" is
the identity transform, and the others raise :class:`NotImplementedError` with
a clear message so the routing layer fails loudly when a route is selected
before the adapter ships. The structure is in place so a follow-up PR can fill
each one in without touching the routing / cache / audit code.

Routing
-------
The integration layer selects an adapter using :func:`select_adapter`. The
selection inputs are:

  - the ``X-Brain-Model-Hint`` request header (takes priority when present)
  - the request body's ``model`` field

If neither names a registered provider, the router defaults to the Anthropic
adapter (identity) so the proxy keeps its existing Claude Code semantics.
"""

from __future__ import annotations

from .base import (
    Adapter,
    AdapterNotImplemented,
    AdapterError,
    RequestIR,
    ResponseIR,
)
from .registry import REGISTRY, register, select_adapter

# Eagerly import known adapter modules so they self-register on the registry.
# The `as foo` aliases keep these intentionally-unused-side-effect imports
# from being flagged as F401 by strict linters.
from . import anthropic as anthropic  # noqa: PLC0414
from . import gemini as gemini  # noqa: PLC0414
from . import ollama as ollama  # noqa: PLC0414
from . import openai as openai  # noqa: PLC0414

__all__ = [
    "REGISTRY",
    "Adapter",
    "AdapterError",
    "AdapterNotImplemented",
    "RequestIR",
    "ResponseIR",
    "register",
    "select_adapter",
]
