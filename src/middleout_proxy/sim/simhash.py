"""64-bit SimHash via signed feature hashing over word shingles.

Pure stdlib. SimHash is a locality-sensitive hash where bitwise Hamming distance
correlates with cosine similarity of the underlying weighted feature vector. We
use unweighted shingles (every shingle contributes ±1) which is the standard
near-duplicate variant.
"""

from __future__ import annotations

import hashlib
import struct

from ..jl import tokenize_words

_BITS = 64
_UINT64_MASK = (1 << _BITS) - 1


def _word_shingles(text: str, shingle_size: int) -> list[bytes]:
    tokens = tokenize_words(text)
    if not tokens:
        return []
    size = max(1, int(shingle_size))
    if len(tokens) <= size:
        return [" ".join(tokens).encode("utf-8")]
    return [
        " ".join(tokens[i : i + size]).encode("utf-8")
        for i in range(len(tokens) - size + 1)
    ]


def _hash64(seed_prefix: bytes, shingle: bytes) -> int:
    digest = hashlib.blake2b(seed_prefix + shingle, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def simhash64(
    text: str,
    *,
    shingle_size: int = 5,
    seed: str = "middleout-sh-v1",
) -> int:
    """Return a 64-bit SimHash for ``text``.

    For every shingle: hash to 64 bits with blake2b. For each bit position
    ``j``: if the bit is set contribute ``+1`` to that position's accumulator,
    otherwise contribute ``-1``. The j-th output bit is 1 iff the accumulator
    is strictly positive.

    Empty / shingle-less text returns ``0``.
    """
    shingles = _word_shingles(text, shingle_size)
    if not shingles:
        return 0

    seed_prefix = seed.encode("utf-8") + b"\x00" + struct.pack("!I", 0)
    counts = [0] * _BITS

    for shingle in shingles:
        h = _hash64(seed_prefix, shingle)
        for j in range(_BITS):
            if (h >> j) & 1:
                counts[j] += 1
            else:
                counts[j] -= 1

    out = 0
    for j in range(_BITS):
        if counts[j] > 0:
            out |= 1 << j
    return out & _UINT64_MASK


def hamming_distance(a: int, b: int) -> int:
    """Population count of ``a ^ b`` restricted to 64 bits."""
    return ((a ^ b) & _UINT64_MASK).bit_count()


def simhash_similarity(a: int, b: int) -> float:
    """``1 - hamming(a, b) / 64`` — a value in ``[0.0, 1.0]``."""
    return 1.0 - hamming_distance(a, b) / _BITS
