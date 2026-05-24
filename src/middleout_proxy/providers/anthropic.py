"""Anthropic adapter — identity transform on the IR.

The proxy's IR *is* the Anthropic Messages schema, so this adapter is the
no-op case: requests pass through, responses pass through. The presence of
this module is what makes the abstraction symmetrical — the routing layer
always selects *some* adapter, and the default is always identity.
"""

from __future__ import annotations

from typing import Any

from .base import RequestIR, ResponseIR
from .registry import register


class AnthropicAdapter:
    """Identity adapter for Anthropic Messages."""

    name: str = "anthropic"
    upstream_base_url: str = "https://api.anthropic.com"

    def translate_request(
        self,
        ir: RequestIR,
    ) -> tuple[str, dict[str, str], bytes]:
        """Identity. The IR is already in Anthropic shape.

        We do drop ``X-Brain-*`` headers on the way out (they were instructions
        to the proxy, not to Anthropic) but otherwise the bytes are the bytes.
        """
        import json

        headers = {
            k: v
            for k, v in ir.headers.items()
            if not k.lower().startswith("x-brain-")
        }
        body = json.dumps(ir.payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return ir.endpoint, headers, body

    def translate_response(
        self,
        *,
        status_code: int,
        headers: dict[str, str],
        body_bytes: bytes,
        media_type: str | None,
    ) -> ResponseIR:
        """Identity. Anthropic's wire format is the IR."""
        import json

        payload: dict[str, Any] | None = None
        if media_type and "application/json" in media_type.lower():
            try:
                decoded = json.loads(body_bytes.decode("utf-8"))
                if isinstance(decoded, dict):
                    payload = decoded
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = None
        return ResponseIR(
            status_code=status_code,
            headers=headers,
            body_bytes=body_bytes,
            payload=payload,
            media_type=media_type,
        )

    def cost_provider(self) -> str:
        return "anthropic"


# Anthropic owns the catch-all default route. We register a broad glob so any
# model starting with ``claude-`` resolves here even before more specific
# routes get added in other adapter modules.
register(AnthropicAdapter(), model_globs=("claude-*",))


__all__ = ["AnthropicAdapter"]
