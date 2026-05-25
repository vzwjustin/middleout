"""LLMLingua-2 wrapper: lazy-loaded in-process prompt compressor for the volatile tail.

LLMLingua-2 (Pan et al., 2024) is a BERT-base token-classification model that
labels each input token as "keep" or "drop"; the surviving tokens form a
lossy-but-semantics-preserving compression of the input. The model file is
~200MB and downloads from HuggingFace on first use.

Design rules
------------
- **Lazy.** The model is imported and loaded only on the first `compress()`
  call. A proxy that never enables LLMLingua-2 must not pay the import cost.
- **Optional.** `llmlingua` and its transitive `transformers`/`torch` deps live
  behind a `[lingua]` install extra. If the deps are missing, `LinguaCompressor`
  raises `LinguaNotInstalled` on first use rather than at import time.
- **Cache-wall safe.** The wrapper is purely a text → text function; it does
  not see the payload structure. The caller (see `volatile.py`) is responsible
  for never feeding it bytes left of the wall.
- **Deterministic given the same model/params.** LLMLingua-2 inference uses
  greedy keep/drop decoding (no sampling) so the output is stable for the same
  model checkpoint and ratio. We pin a known checkpoint and document the pin.
- **Token-count threshold.** Very short inputs (< 64 tokens) are passed through
  unchanged — the compressor's labelling overhead outweighs any savings, and
  short blocks are usually conversational glue that doesn't compress well.

Failure modes
-------------
- Model download fails → `LinguaUnavailable` (transient, callers may retry).
- Compression produces output longer than input → return the input verbatim.
- Inference raises → return the input verbatim, record a warning event.

The compressor is intentionally fail-soft on the proxy path: a broken
LLMLingua-2 must not break the proxy, it must just stop saving tokens.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


_DEFAULT_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
_DEFAULT_RATIO = 0.5  # keep ~50% of tokens; tunable per request
_MIN_TOKEN_THRESHOLD = 64
_MAX_INPUT_CHARS = 200_000  # refuse pathologically large inputs to keep latency bounded


class LinguaNotInstalled(RuntimeError):
    """The optional [lingua] install extra is not present.

    Install with: `pip install 'middleout-claude-proxy[lingua]'` (or add
    `llmlingua` + `transformers` + `torch` to your environment).
    """


class LinguaUnavailable(RuntimeError):
    """LLMLingua-2 deps are present but the model cannot be loaded right now.

    Transient — usually means the HuggingFace download failed or the local
    cache is corrupt. Callers may retry, or fall back to the existing engines.
    """


@dataclass
class LinguaResult:
    """Result of one compression call.

    `chars_in/chars_out` are byte-naive Python string lengths (not tokens) —
    matches the proxy's existing `chars_saved` accounting across audit logs and
    the dashboard. `dropped_token_count` is the token-level estimate from
    LLMLingua-2 itself when available.
    """

    text: str
    chars_in: int
    chars_out: int
    dropped_token_count: int = 0
    skipped_reason: str | None = None

    @property
    def chars_saved(self) -> int:
        return max(0, self.chars_in - self.chars_out)

    @property
    def changed(self) -> bool:
        return self.text != self._original or (self.skipped_reason is None and self.chars_saved > 0)

    # `_original` is set by the compressor when constructing the result so that
    # `changed` can answer correctly even when chars_in == chars_out (e.g., the
    # model emitted exactly the input back).
    _original: str = ""


class LinguaCompressor:
    """Thread-safe lazy-loaded LLMLingua-2 wrapper.

    Construction is cheap — no model is loaded until the first `compress()`
    call. Subsequent calls reuse the same in-memory model. The instance is
    safe to share across asyncio tasks (the underlying model call is wrapped
    in a lock to avoid concurrent inference on the same model object).
    """

    def __init__(
        self,
        *,
        model_id: str = _DEFAULT_MODEL,
        default_ratio: float = _DEFAULT_RATIO,
        device: str | None = None,
    ) -> None:
        if not 0.05 <= default_ratio <= 0.95:
            raise ValueError(
                f"default_ratio must be in [0.05, 0.95], got {default_ratio}"
            )
        self.model_id = model_id
        self.default_ratio = default_ratio
        self.device = device  # None lets llmlingua pick (cpu/cuda/mps)
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._load_attempted = False
        self._load_error: BaseException | None = None

    def _ensure_model(self) -> Any:
        """Load the model on first use. Re-raises load failures on every call.

        We deliberately do NOT retry indefinitely on load failure — if HuggingFace
        is unreachable, subsequent calls should fail fast so the proxy can fall
        back to other engines instead of hanging on every request.
        """
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            if self._load_attempted and self._load_error is not None:
                # Subsequent failure — surface the original error.
                raise self._load_error

            self._load_attempted = True
            try:
                from llmlingua import PromptCompressor  # type: ignore[import-not-found]
            except ImportError as e:
                err = LinguaNotInstalled(
                    "llmlingua is not installed. Install the [lingua] extra: "
                    "`pip install 'middleout-claude-proxy[lingua]'`."
                )
                err.__cause__ = e
                self._load_error = err
                raise err from e

            try:
                kwargs: dict[str, Any] = {
                    "model_name": self.model_id,
                    "use_llmlingua2": True,
                }
                if self.device is not None:
                    kwargs["device_map"] = self.device
                self._model = PromptCompressor(**kwargs)
            except Exception as e:
                err = LinguaUnavailable(
                    f"Could not load LLMLingua-2 model {self.model_id!r}: "
                    f"{type(e).__name__}: {e}"
                )
                err.__cause__ = e
                self._load_error = err
                raise err from e

        return self._model

    def compress(self, text: str, *, ratio: float | None = None) -> LinguaResult:
        """Compress `text` to roughly `ratio` of its original token count.

        Pass-through cases (return `text` verbatim with a `skipped_reason`):
        - Input is empty or shorter than `_MIN_TOKEN_THRESHOLD` tokens (we use a
          ~4-chars/token rule of thumb to avoid loading the tokenizer for the
          decision).
        - Input is larger than `_MAX_INPUT_CHARS` — we refuse rather than risk
          a multi-second inference.
        - The compressor errors mid-inference.
        - The compressor returns output >= the input length (no win).
        """
        original = text
        chars_in = len(text)

        if chars_in == 0:
            return LinguaResult("", 0, 0, skipped_reason="empty", _original=original)

        if chars_in < _MIN_TOKEN_THRESHOLD * 4:
            return LinguaResult(
                text, chars_in, chars_in, skipped_reason="too_small", _original=original
            )

        if chars_in > _MAX_INPUT_CHARS:
            return LinguaResult(
                text, chars_in, chars_in, skipped_reason="too_large", _original=original
            )

        eff_ratio = ratio if ratio is not None else self.default_ratio
        if not 0.05 <= eff_ratio <= 0.95:
            raise ValueError(f"ratio must be in [0.05, 0.95], got {eff_ratio}")

        try:
            model = self._ensure_model()
        except (LinguaNotInstalled, LinguaUnavailable) as e:
            logger.warning("LLMLingua-2 unavailable: %s", e)
            return LinguaResult(
                text, chars_in, chars_in, skipped_reason="unavailable", _original=original
            )

        try:
            with self._model_lock:
                out = model.compress_prompt(
                    text,
                    rate=eff_ratio,
                    force_tokens=["\n", "?", "!", "."],
                )
        except Exception as e:
            logger.warning("LLMLingua-2 inference failed: %s: %s", type(e).__name__, e)
            return LinguaResult(
                text, chars_in, chars_in, skipped_reason="inference_error", _original=original
            )

        compressed = _extract_compressed_text(out)
        if not isinstance(compressed, str) or len(compressed) >= chars_in:
            return LinguaResult(
                text, chars_in, chars_in, skipped_reason="no_win", _original=original
            )

        dropped = 0
        if isinstance(out, dict):
            try:
                dropped = int(out.get("origin_tokens", 0)) - int(out.get("compressed_tokens", 0))
            except (TypeError, ValueError):
                dropped = 0

        return LinguaResult(
            compressed,
            chars_in,
            len(compressed),
            dropped_token_count=max(0, dropped),
            _original=original,
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def reset(self) -> None:
        """Drop the loaded model. Next compress() will re-load."""
        with self._model_lock:
            self._model = None
            self._load_attempted = False
            self._load_error = None


def _extract_compressed_text(out: Any) -> str | None:
    """LLMLingua-2's return shape changed between versions; handle both.

    Older: returns a plain string.
    Newer: returns a dict with `compressed_prompt` key.
    """
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        for key in ("compressed_prompt", "compressed_text", "text"):
            val = out.get(key)
            if isinstance(val, str):
                return val
    return None


__all__ = [
    "LinguaCompressor",
    "LinguaNotInstalled",
    "LinguaResult",
    "LinguaUnavailable",
]
