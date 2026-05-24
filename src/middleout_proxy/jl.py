from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[^\s]")


def tokenize(text: str) -> list[str]:
    """Cheap tokenizer good enough for local similarity sketches."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def shingles(tokens: list[str], width: int) -> Iterable[str]:
    if not tokens:
        return
    width = max(1, width)
    if len(tokens) <= width:
        yield " ".join(tokens)
        return
    for i in range(0, len(tokens) - width + 1):
        yield " ".join(tokens[i : i + width])


def _hash64(seed: str, value: str) -> int:
    digest = hashlib.blake2b(f"{seed}\0{value}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def signed_jl_projection(
    text: str,
    *,
    dims: int = 512,
    shingle_tokens: int = 5,
    seed: str = "middleout-jl-v1",
) -> tuple[float, ...]:
    """Return a normalized random sign-projection sketch of text shingles.

    This is a practical feature-hashing/random-projection sketch inspired by the
    Johnson-Lindenstrauss family of transforms. It is not a reversible encoding.
    """
    vec = [0.0] * dims
    toks = tokenize(text)
    for shingle in shingles(toks, shingle_tokens):
        h = _hash64(seed, shingle)
        idx = h % dims
        sign = 1.0 if ((h >> 32) & 1) else -1.0
        vec[idx] += sign

    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return tuple(vec)
    return tuple(v / norm for v in vec)


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimensionality")
    return sum(x * y for x, y in zip(a, b))


@dataclass
class SketchRecord:
    path: str
    digest: str
    chars: int
    sketch: tuple[float, ...]


class RequestSketchIndex:
    """In-memory index for one request body."""

    def __init__(self, *, dims: int, shingle_tokens: int) -> None:
        self.dims = dims
        self.shingle_tokens = shingle_tokens
        self._records: list[SketchRecord] = []

    def find_best(self, text: str) -> tuple[SketchRecord | None, float]:
        sketch = signed_jl_projection(text, dims=self.dims, shingle_tokens=self.shingle_tokens)
        best_record: SketchRecord | None = None
        best_score = -1.0
        for record in self._records:
            score = cosine(sketch, record.sketch)
            if score > best_score:
                best_score = score
                best_record = record
        return best_record, best_score

    def add(self, *, text: str, path: str, digest: str) -> None:
        sketch = signed_jl_projection(text, dims=self.dims, shingle_tokens=self.shingle_tokens)
        self._records.append(SketchRecord(path=path, digest=digest, chars=len(text), sketch=sketch))
