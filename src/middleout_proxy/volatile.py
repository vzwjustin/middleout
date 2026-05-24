"""Volatile-tail compressor: applies LLMLingua-2 only to blocks right of the wall.

The volatile compressor is the proxy-side glue between `cache_wall` (which
decides what's protected) and `lingua` (which knows how to shrink a text
block). It walks message content blocks, consults the wall, hands off each
unprotected text block to LLMLingua-2, and writes the result back in place.

Hard rules
----------
1. Never even *read* a block left of the wall when running with `auto_insert=False`.
   When `auto_insert=True`, the inserted breakpoint is the *first* thing in the
   tail-volatile region by construction, so the same rule holds.
2. Never serialize the protected prefix differently than how the client sent
   it. We deepcopy the payload at entry and only mutate volatile block dicts.
3. Fail soft: any unexpected block shape (non-string text, unknown type) is
   skipped, never crashes the request.
4. Audit every change. The compressor returns a list of `VolatileEvent` so
   the audit log and the dashboard can attribute savings per block.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .cache_wall import CacheWall, iter_volatile_blocks
from .lingua import LinguaCompressor, LinguaResult


@dataclass
class VolatileEvent:
    """One block-level compression record.

    `path` is a human-readable location like `messages[3].content[0]` for the
    audit log; `kind` is the cache_wall block kind. `chars_in/out` and
    `skipped_reason` are passed through from `LinguaResult` so the event
    captures why no savings happened when applicable.
    """

    path: str
    kind: str
    msg_idx: int | None
    block_idx: int
    chars_in: int
    chars_out: int
    skipped_reason: str | None = None

    @property
    def chars_saved(self) -> int:
        return max(0, self.chars_in - self.chars_out)


@dataclass
class VolatileAudit:
    """Aggregate audit for one request's worth of volatile-tail compression."""

    events: list[VolatileEvent] = field(default_factory=list)
    blocks_inspected: int = 0
    blocks_skipped_protected: int = 0
    blocks_skipped_non_text: int = 0

    @property
    def chars_saved(self) -> int:
        return sum(e.chars_saved for e in self.events)

    @property
    def touched(self) -> bool:
        return any(e.chars_saved > 0 for e in self.events)


# Tool-result blocks usually carry the most volume of compressible text in
# real Claude Code traffic — they're the file reads, the search results, the
# command outputs. User text and assistant text are typically already short.
_COMPRESSIBLE_BLOCK_TYPES = frozenset({"text", "tool_result", "tool_use"})


def compress_volatile_tail(
    payload: dict[str, Any],
    *,
    wall: CacheWall,
    lingua: LinguaCompressor,
    ratio: float | None = None,
    deepcopy_payload: bool = True,
) -> tuple[dict[str, Any], VolatileAudit]:
    """Compress every volatile text block in `payload` via `lingua`.

    Returns (new_payload, audit). The original `payload` is left untouched when
    `deepcopy_payload=True` (default — safest for callers that may retry the
    raw bytes on upstream failure). With `deepcopy_payload=False` the caller
    accepts in-place mutation in exchange for less GC pressure on large
    payloads.

    When the wall has no marker (whole payload is volatile) and `lingua` is
    unavailable, this still returns `(payload, audit)` with `audit.touched=False`
    — the proxy can fall back to other engines.
    """
    audit = VolatileAudit()
    target = copy.deepcopy(payload) if deepcopy_payload else payload

    for kind, msg_idx, block_idx, block in iter_volatile_blocks(target, wall):
        audit.blocks_inspected += 1

        text, write_back = _extract_text_for_compression(block)
        if text is None or write_back is None:
            audit.blocks_skipped_non_text += 1
            continue

        result: LinguaResult = lingua.compress(text, ratio=ratio)
        path = _format_block_path(kind, msg_idx, block_idx)

        if result.text is text or result.chars_out >= result.chars_in:
            # No-op — still record the inspection so the dashboard can show
            # "blocks considered but skipped" if we ever want to.
            audit.events.append(
                VolatileEvent(
                    path=path,
                    kind=kind,
                    msg_idx=msg_idx,
                    block_idx=block_idx,
                    chars_in=result.chars_in,
                    chars_out=result.chars_out,
                    skipped_reason=result.skipped_reason or "no_win",
                )
            )
            continue

        write_back(target, msg_idx, block_idx, result.text)
        audit.events.append(
            VolatileEvent(
                path=path,
                kind=kind,
                msg_idx=msg_idx,
                block_idx=block_idx,
                chars_in=result.chars_in,
                chars_out=result.chars_out,
                skipped_reason=result.skipped_reason,
            )
        )

    return target, audit


def _format_block_path(kind: str, msg_idx: int | None, block_idx: int) -> str:
    if kind == "message":
        return f"messages[{msg_idx}].content[{block_idx}]"
    return f"{kind}[{block_idx}]"


def _extract_text_for_compression(block: Any) -> tuple[str | None, Any]:
    """Return (text, write_back) for compressible blocks, else (None, None).

    `write_back(payload, msg_idx, block_idx, new_text)` mutates the payload to
    install `new_text` at the right position. Different block types need
    different write-back logic; we hand a closure back so the caller doesn't
    need to switch on type.
    """
    # iter_volatile_blocks yields plain strings for string-shaped message
    # content. We must rewrite the whole message.content to the new string.
    if isinstance(block, str):
        def _write_back_str(payload: dict[str, Any], mi: int | None, bi: int, new_text: str) -> None:
            if mi is None:
                return
            payload["messages"][mi]["content"] = new_text
        return block, _write_back_str

    if not isinstance(block, dict):
        return None, None

    btype = block.get("type")
    if btype not in _COMPRESSIBLE_BLOCK_TYPES:
        return None, None

    if btype == "text":
        text = block.get("text")
        if not isinstance(text, str):
            return None, None
        def _write_back_text(payload: dict[str, Any], mi: int | None, bi: int, new_text: str) -> None:
            if mi is None:
                return
            payload["messages"][mi]["content"][bi]["text"] = new_text
        return text, _write_back_text

    if btype == "tool_result":
        # tool_result content can be a string or a list of {type:"text",text:...}
        content = block.get("content")
        if isinstance(content, str):
            def _write_back_tr_str(payload: dict[str, Any], mi: int | None, bi: int, new_text: str) -> None:
                if mi is None:
                    return
                payload["messages"][mi]["content"][bi]["content"] = new_text
            return content, _write_back_tr_str
        if isinstance(content, list):
            # Pick the largest text sub-block to compress. (We deliberately do
            # not compress every sub-block in one shot — keeps the audit
            # event-per-block invariant and the cache-wall logic clean.)
            best_idx = -1
            best_len = 0
            for i, sub in enumerate(content):
                if isinstance(sub, dict) and sub.get("type") == "text":
                    t = sub.get("text")
                    if isinstance(t, str) and len(t) > best_len:
                        best_idx, best_len = i, len(t)
            if best_idx < 0:
                return None, None
            chosen_text = content[best_idx]["text"]
            def _write_back_tr_list(payload: dict[str, Any], mi: int | None, bi: int, new_text: str) -> None:
                if mi is None:
                    return
                payload["messages"][mi]["content"][bi]["content"][best_idx]["text"] = new_text
            return chosen_text, _write_back_tr_list

    # tool_use blocks have a JSON-ish "input" field; not safe to compress
    # since the LLM may have generated structured args. Skip.
    return None, None


__all__ = [
    "VolatileAudit",
    "VolatileEvent",
    "compress_volatile_tail",
]
