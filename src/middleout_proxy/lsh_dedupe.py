"""LSH-based dedupe: minhash + banded LSH to find near-duplicate blocks.

Pure-stdlib. Deterministic. Uses blake2b for minhash so the same input always
produces the same signature.

Public API:
  LSHDedupeIndex(level: str)
      .add(block_id, text)
      .find_near_duplicate(text) -> Optional[(block_id, similarity)]
  dedupe_blocks(blocks: list[dict], level: str, protected: set[int])
      -> tuple[list[dict], dict]

Levels:
  - conservative : threshold 0.95, bands b=8, rows r=16  (signature width 128)
  - standard     : threshold 0.88, bands b=16, rows r=8
  - aggressive   : threshold 0.80, bands b=32, rows r=4

`protected` is a set of block indices that must be left untouched. They still
contribute to the index so later candidates can match against them.
"""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Optional

from .jl import shingles, tokenize

_LEVELS = ("conservative", "standard", "aggressive")

_SIGNATURE_WIDTH = 128
_SHINGLE_WIDTH = 5

_LEVEL_CONFIG = {
    "conservative": {"threshold": 0.95, "bands": 8, "rows": 16},
    "standard":     {"threshold": 0.88, "bands": 16, "rows": 8},
    "aggressive":   {"threshold": 0.80, "bands": 32, "rows": 4},
}

# Maximum 64-bit value used as the "infinity" for unseen hash buckets.
_MAX_U64 = (1 << 64) - 1


def _hash_token(seed_idx: int, token: str) -> int:
    """Deterministic 64-bit blake2b hash, parameterized by seed index."""
    h = hashlib.blake2b(
        token.encode("utf-8", errors="replace"),
        digest_size=8,
        person=f"mh{seed_idx:04d}".encode("ascii"),
    ).digest()
    return int.from_bytes(h, "big", signed=False)


def _minhash_signature(text: str, width: int = _SIGNATURE_WIDTH) -> tuple[int, ...]:
    """Deterministic minhash signature over k-shingles of text."""
    tokens = tokenize(text)
    sigs = [_MAX_U64] * width
    seen = False
    for shingle in shingles(tokens, _SHINGLE_WIDTH):
        seen = True
        for i in range(width):
            h = _hash_token(i, shingle)
            if h < sigs[i]:
                sigs[i] = h
    if not seen:
        # Empty/very short input — return all zeros so comparisons are stable.
        return tuple([0] * width)
    return tuple(sigs)


def _band_keys(signature: tuple[int, ...], bands: int, rows: int) -> list[bytes]:
    """Split signature into `bands` groups of `rows` minhashes; hash each group to a key."""
    keys: list[bytes] = []
    for b in range(bands):
        chunk = signature[b * rows: (b + 1) * rows]
        # Pack as bytes then digest; deterministic.
        packed = b"".join(v.to_bytes(8, "big", signed=False) for v in chunk)
        keys.append(hashlib.blake2b(packed, digest_size=8).digest())
    return keys


def _jaccard_estimate(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / len(a)


class LSHDedupeIndex:
    """Banded-LSH minhash index for near-duplicate detection within one request."""

    def __init__(self, level: str = "standard") -> None:
        if level not in _LEVELS:
            raise ValueError(f"lsh level must be one of {_LEVELS}, got {level!r}")
        self.level = level
        cfg = _LEVEL_CONFIG[level]
        self.threshold: float = cfg["threshold"]
        self.bands: int = cfg["bands"]
        self.rows: int = cfg["rows"]
        # bands * rows must equal signature width.
        assert self.bands * self.rows == _SIGNATURE_WIDTH
        # band_idx -> band_key -> list[block_id]
        self._bands: list[dict[bytes, list[object]]] = [dict() for _ in range(self.bands)]
        self._signatures: dict[object, tuple[int, ...]] = {}

    def add(self, block_id: object, text: str) -> None:
        sig = _minhash_signature(text)
        self._signatures[block_id] = sig
        keys = _band_keys(sig, self.bands, self.rows)
        for b, key in enumerate(keys):
            self._bands[b].setdefault(key, []).append(block_id)

    def find_near_duplicate(self, text: str) -> Optional[tuple[object, float]]:
        """Return (block_id, similarity) of the best candidate >= threshold, else None."""
        if not self._signatures:
            return None
        sig = _minhash_signature(text)
        keys = _band_keys(sig, self.bands, self.rows)
        candidates: set[object] = set()
        for b, key in enumerate(keys):
            bucket = self._bands[b].get(key)
            if bucket:
                candidates.update(bucket)
        best: tuple[object, float] | None = None
        for cand in candidates:
            other = self._signatures.get(cand)
            if other is None:
                continue
            score = _jaccard_estimate(sig, other)
            if score >= self.threshold and (best is None or score > best[1]):
                best = (cand, score)
        return best


_TEXT_RE = re.compile(r"[A-Za-z0-9]")


def _block_text(block: dict) -> str | None:
    """Extract the text payload of a content block for dedupe purposes."""
    if not isinstance(block, dict):
        return None
    btype = block.get("type")
    if btype == "text" and isinstance(block.get("text"), str):
        return block["text"]
    if btype == "tool_result":
        content = block.get("content")
        if isinstance(content, str):
            return content
    return None


def _set_block_text(block: dict, new_text: str) -> None:
    btype = block.get("type")
    if btype == "text":
        block["text"] = new_text
    elif btype == "tool_result":
        block["content"] = new_text


def dedupe_blocks(
    blocks: list[dict],
    level: str = "standard",
    protected: set[int] | None = None,
) -> tuple[list[dict], dict]:
    """Replace near-duplicate later blocks with a marker referencing the earlier one.

    Returns (new_blocks, stats). `protected` is a set of indices to leave alone;
    protected blocks may still serve as the "earlier" match for later ones.

    Stats: {"replaced": int, "level": str, "threshold": float, "protected": int}
    """
    if level not in _LEVELS:
        raise ValueError(f"lsh level must be one of {_LEVELS}, got {level!r}")
    if protected is None:
        protected = set()

    new_blocks = copy.deepcopy(blocks)
    index = LSHDedupeIndex(level=level)
    replaced = 0
    for i, block in enumerate(new_blocks):
        text = _block_text(block)
        if not isinstance(text, str) or not text:
            continue
        if i in protected:
            index.add(i, text)
            continue
        match = index.find_near_duplicate(text)
        if match is not None:
            other_id, score = match
            marker = (
                f"[duplicate of earlier block at {other_id}, ~{len(text)} chars, "
                f"similarity {score:.2f}]"
            )
            _set_block_text(block, marker)
            replaced += 1
            # We do NOT add the replaced block back to the index — only originals
            # contribute, so later near-dups still match the first occurrence.
        else:
            index.add(i, text)

    stats = {
        "replaced": replaced,
        "level": level,
        "threshold": _LEVEL_CONFIG[level]["threshold"],
        "protected": len(protected),
    }
    return new_blocks, stats
