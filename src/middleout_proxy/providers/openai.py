"""OpenAI adapter scaffold.

Not yet implemented end-to-end. The class is registered so the routing layer
can select it for `gpt-*` models, but `translate_request` / `translate_response`
raise :class:`AdapterNotImplemented` until the body-shape translation is
filled in. Wiring this means:

  - Anthropic ``messages`` → OpenAI ``messages`` (roles align, content blocks
    flatten to strings, tool_use ↔ tool_calls).
  - Anthropic ``system`` (string or list) → OpenAI system message at index 0.
  - Anthropic ``tools`` (JSON schema) → OpenAI ``tools`` (subset of the same
    schema). Anthropic uses ``input_schema``; OpenAI uses ``parameters``.
  - Streaming events: OpenAI ``data: ...`` JSON deltas → Anthropic ``content_block_delta``
    / ``message_delta``. Non-trivial; the first PR can implement non-streaming
    only and 501 on streaming requests.

A real implementation lives behind this scaffold so the integration layer is
already wiring the right shape into the cache and the cost tracker.
"""

from __future__ import annotations

from .base import AdapterNotImplemented, RequestIR, ResponseIR
from .registry import register


class OpenAIAdapter:
    name: str = "openai"
    upstream_base_url: str = "https://api.openai.com/v1"

    def translate_request(self, ir: RequestIR) -> tuple[str, dict[str, str], bytes]:
        raise AdapterNotImplemented(
            "OpenAI adapter request translation is not implemented yet. "
            "Route X-Brain-Model-Hint to 'anthropic' until this ships, "
            "or open a PR — base shape is in providers/openai.py."
        )

    def translate_response(
        self,
        *,
        status_code: int,
        headers: dict[str, str],
        body_bytes: bytes,
        media_type: str | None,
    ) -> ResponseIR:
        raise AdapterNotImplemented(
            "OpenAI adapter response translation is not implemented yet."
        )

    def cost_provider(self) -> str:
        return "openai"


register(OpenAIAdapter(), model_globs=("gpt-*", "o1-*", "o3-*", "openai/*"))


__all__ = ["OpenAIAdapter"]
