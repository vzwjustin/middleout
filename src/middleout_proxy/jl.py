from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from collections.abc import Iterable

__all__ = [
    "BandedJLIndex",
    "RequestSketchIndex",
    "SketchRecord",
    "cosine",
    "shingles",
    "signed_jl_projection",
    "tokenize",
    "tokenize_words",
]

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[^\s]")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Cheap tokenizer good enough for local similarity sketches."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def tokenize_words(text: str) -> list[str]:
    """Alphanumeric-only word tokenizer (lowercased). Used by MinHash and SimHash.

    Strips all punctuation/whitespace. Returns the lowercased contiguous
    alphanumeric runs in order.
    """
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


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
    digest = hashlib.blake2b(f"{seed}\0{value}".encode(), digest_size=8).digest()
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
    return sum(x * y for x, y in zip(a, b, strict=True))


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


class BandedJLIndex:
    """JL sign-projection sketch index accelerated by MinHash banded LSH.

    Public surface mirrors :class:`RequestSketchIndex` so consumers can swap in
    place. ``find_best`` first asks the LSH for candidate doc ids, then ranks
    those candidates by JL cosine. Falls back to a full scan when the corpus is
    very small (``< small_corpus_cutoff``) or when the LSH yields nothing.
    """

    def __init__(
        self,
        *,
        dims: int,
        shingle_tokens: int,
        mh_num_perms: int = 128,
        mh_bands: int = 32,
        mh_shingle_size: int = 5,
        small_corpus_cutoff: int = 16,
    ) -> None:
        # Local import avoids a circular dependency with ``sim`` modules that
        # themselves import from ``jl``.
        from .sim.lsh import MinHashLSH
        from .sim.minhash import minhash_signature

        self.dims = dims
        self.shingle_tokens = shingle_tokens
        self.mh_num_perms = mh_num_perms
        self.mh_bands = mh_bands
        self.mh_shingle_size = mh_shingle_size
        self.small_corpus_cutoff = small_corpus_cutoff
        self._records: list[SketchRecord] = []
        self._record_by_id: dict[str, SketchRecord] = {}
        self._lsh = MinHashLSH(num_perms=mh_num_perms, bands=mh_bands)
        self._minhash = minhash_signature

    def add(self, *, text: str, path: str, digest: str) -> None:
        sketch = signed_jl_projection(text, dims=self.dims, shingle_tokens=self.shingle_tokens)
        record = SketchRecord(path=path, digest=digest, chars=len(text), sketch=sketch)
        doc_id = f"{path}#{digest}#{len(self._records)}"
        self._records.append(record)
        self._record_by_id[doc_id] = record
        sig = self._minhash(
            text, num_perms=self.mh_num_perms, shingle_size=self.mh_shingle_size
        )
        self._lsh.add(doc_id, sig)

    def find_best(self, text: str) -> tuple[SketchRecord | None, float]:
        if not self._records:
            return None, -1.0
        sketch = signed_jl_projection(
            text, dims=self.dims, shingle_tokens=self.shingle_tokens
        )

        if len(self._records) < self.small_corpus_cutoff:
            return self._scan(sketch, self._records)

        sig = self._minhash(
            text, num_perms=self.mh_num_perms, shingle_size=self.mh_shingle_size
        )
        candidate_ids = self._lsh.candidates(sig)
        if not candidate_ids:
            return None, -1.0

        candidates = [self._record_by_id[c] for c in candidate_ids if c in self._record_by_id]
        if not candidates:
            return None, -1.0
        return self._scan(sketch, candidates)

    def _scan(
        self,
        sketch: tuple[float, ...],
        records: list[SketchRecord],
    ) -> tuple[SketchRecord | None, float]:
        best_record: SketchRecord | None = None
        best_score = -1.0
        for record in records:
            score = cosine(sketch, record.sketch)
            if score > best_score:
                best_score = score
                best_record = record
        return best_record, best_score

    def __len__(self) -> int:
        return len(self._records)
