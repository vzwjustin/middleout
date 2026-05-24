"""Google Gemini adapter scaffold.

Routes ``gemini-*`` model ids here. The translation work is similar to the
OpenAI adapter but with Google's ``contents`` / ``parts`` shape and the
``generationConfig`` block. Streaming format is a server-sent event of JSON
deltas; non-streaming is one JSON blob.

Not yet implemented end-to-end.
"""

from __future__ import annotations

from .base import AdapterNotImplemented, RequestIR, ResponseIR
from .registry import register


class GeminiAdapter:
    name: str = "google"
    upstream_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    def translate_request(self, ir: RequestIR) -> tuple[str, dict[str, str], bytes]:
        raise AdapterNotImplemented(
            "Gemini adapter request translation is not implemented yet."
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
            "Gemini adapter response translation is not implemented yet."
        )

    def cost_provider(self) -> str:
        return "google"


register(GeminiAdapter(), model_globs=("gemini-*", "google/*"))


__all__ = ["GeminiAdapter"]
