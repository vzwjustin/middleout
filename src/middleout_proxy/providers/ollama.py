"""Ollama / local-model adapter scaffold.

Routes ``ollama:*`` or ``local/*`` ids here. The upstream is whatever Ollama
instance is reachable on the proxy host (default ``http://127.0.0.1:11434``).

Not yet implemented end-to-end. Ollama exposes both an OpenAI-compatible
endpoint (``/v1/chat/completions``) and a native one (``/api/chat``); the
adapter should prefer the OpenAI-compatible surface for code reuse with the
OpenAI adapter when that ships.
"""

from __future__ import annotations

from .base import AdapterNotImplemented, RequestIR, ResponseIR
from .registry import register


class OllamaAdapter:
    name: str = "ollama"
    upstream_base_url: str = "http://127.0.0.1:11434"

    def translate_request(self, ir: RequestIR) -> tuple[str, dict[str, str], bytes]:
        raise AdapterNotImplemented(
            "Ollama adapter request translation is not implemented yet."
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
            "Ollama adapter response translation is not implemented yet."
        )

    def cost_provider(self) -> str:
        return "ollama"


register(
    OllamaAdapter(),
    model_globs=("ollama:*", "ollama/*", "local/*"),
)


__all__ = ["OllamaAdapter"]
