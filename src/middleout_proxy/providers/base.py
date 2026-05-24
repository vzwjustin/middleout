"""Adapter base types and IR (intermediate representation).

The IR is the Anthropic Messages schema. We model it as a thin wrapper around
a `dict` (rather than a full pydantic model) so the proxy can pass through
fields it doesn't know about without rejecting them. Forward compatibility
beats type safety in a proxy layer.

Adapters are stateless. Construction is cheap; the runtime keeps one
instance per provider in :data:`providers.registry.REGISTRY`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class AdapterError(Exception):
    """Base for adapter-translation failures."""


class AdapterNotImplemented(AdapterError):
    """Raised when an adapter exists in name but hasn't been implemented yet.

    Routes that resolve to a not-yet-implemented adapter should surface this
    as a 501 to the client so the failure is unambiguous.
    """


@dataclass
class RequestIR:
    """Outgoing request in the canonical (Anthropic) representation.

    ``payload`` is the raw JSON body the client sent (already
    cache-wall-protected and compressed). ``headers`` is the forwardable
    subset (auth + anthropic-version + anything the strict-header filter let
    through).

    ``model_hint`` is the optional ``X-Brain-Model-Hint`` override — when set,
    the router prefers it over ``payload['model']`` for adapter selection.
    """

    payload: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    model_hint: str | None = None
    endpoint: str = "v1/messages"

    @property
    def model(self) -> str:
        if isinstance(self.payload, dict):
            m = self.payload.get("model")
            if isinstance(m, str):
                return m
        return ""

    @property
    def effective_model(self) -> str:
        """The model id used for routing. Hint wins over body."""
        return self.model_hint or self.model


@dataclass
class ResponseIR:
    """Upstream response coerced back to the Anthropic Messages shape.

    ``status_code``, ``headers``, and ``body_bytes`` are the verbatim bytes
    that should be relayed to the client. Adapters that translate from a
    non-Anthropic provider construct both the IR ``payload`` (for downstream
    bookkeeping like cost) and the byte body the client sees.
    """

    status_code: int
    headers: dict[str, str]
    body_bytes: bytes
    payload: dict[str, Any] | None = None
    media_type: str | None = None


class Adapter(Protocol):
    """Adapter interface.

    Implementations live in sibling modules in this package. They are
    intentionally narrow — translate request, translate response, advertise a
    name. Routing, caching, and audit happen one layer up so the adapter has
    nothing to know about HTTP, timeouts, or compression.
    """

    name: str
    upstream_base_url: str  # default upstream when the integration layer doesn't override

    def translate_request(self, ir: RequestIR) -> tuple[str, dict[str, str], bytes]:
        """Convert IR to ``(url_path, headers, body_bytes)`` ready for upstream.

        ``url_path`` is appended to the adapter's base URL. The body bytes are
        the wire-format request the upstream expects.
        """

    def translate_response(
        self,
        *,
        status_code: int,
        headers: dict[str, str],
        body_bytes: bytes,
        media_type: str | None,
    ) -> ResponseIR:
        """Convert an upstream response to the canonical IR.

        Identity for Anthropic. For other providers, this also re-shapes the
        body bytes so the downstream client sees an Anthropic Messages
        response.
        """

    def cost_provider(self) -> str:
        """The provider key used by :mod:`cost` to look up pricing."""


def _filter_anthropic_headers(headers: dict[str, str]) -> dict[str, str]:
    """Anthropic-only set of headers we know are safe to forward (post-strict-filter).

    The strict header filter in `server._forward_request_headers` already drops
    api-key style auth; this is the conservative second layer that keeps only
    well-known Anthropic-shaped headers. Adapters that need vendor-specific
    headers compose their own dict in ``translate_request``.
    """
    keep_prefixes = ("anthropic-", "x-claude-", "x-brain-")
    keep_exact = {"authorization", "content-type", "user-agent", "accept"}
    out: dict[str, str] = {}
    for k, v in headers.items():
        lk = k.lower()
        if lk in keep_exact or any(lk.startswith(p) for p in keep_prefixes):
            out[lk] = v
    return out


__all__ = [
    "Adapter",
    "AdapterError",
    "AdapterNotImplemented",
    "RequestIR",
    "ResponseIR",
    "_filter_anthropic_headers",
]
