"""MinHash signatures over alphanumeric word shingles.

Pure stdlib. The i-th permutation is realised by prefixing every shingle with
``seed || NUL || pack("!I", i)`` before hashing with blake2b. The signature at
position ``i`` is the minimum 64-bit digest over every shingle for that
permutation — classic MinHash with ``num_perms`` independent hash families.
"""

from __future__ import annotations

import hashlib
import struct

from ..jl import tokenize_words

_UINT64_MAX = (1 << 64) - 1


def _word_shingles(text: str, shingle_size: int) -> list[bytes]:
    """Return k-shingles over alphanumeric word tokens, as UTF-8 bytes.

    Returning bytes (not str) lets us feed them straight into ``hashlib`` and
    keeps the hot loop tight. Each shingle is the space-joined run of the
    underlying tokens, encoded as UTF-8.
    """
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


def minhash_signature(
    text: str,
    *,
    num_perms: int = 128,
    shingle_size: int = 5,
    seed: str = "middleout-mh-v1",
) -> tuple[int, ...]:
    """Return a MinHash signature of length ``num_perms`` over word k-shingles.

    Empty / shingle-less text returns ``(2**64 - 1,) * num_perms`` so that
    :func:`jaccard_estimate` against any real signature is 0.0 (except in the
    trivial case of two empty docs, where it is 1.0 by definition).
    """
    if num_perms <= 0:
        raise ValueError("num_perms must be positive")

    shingle_bytes = _word_shingles(text, shingle_size)
    if not shingle_bytes:
        return (_UINT64_MAX,) * num_perms

    seed_prefix = seed.encode("utf-8") + b"\x00"
    perm_prefixes = [seed_prefix + struct.pack("!I", i) for i in range(num_perms)]
    mins = [_UINT64_MAX] * num_perms

    for i in range(num_perms):
        prefix = perm_prefixes[i]
        local_min = _UINT64_MAX
        for shingle in shingle_bytes:
            digest = hashlib.blake2b(prefix + shingle, digest_size=8).digest()
            value = int.from_bytes(digest, "big", signed=False)
            if value < local_min:
                local_min = value
        mins[i] = local_min

    return tuple(mins)


def jaccard_estimate(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    """Fraction of matching MinHash positions ≈ Jaccard similarity.

    Raises ``ValueError`` on dimension mismatch — silently mismatched
    signatures would otherwise produce meaningless scores.
    """
    if len(sig_a) != len(sig_b):
        raise ValueError(
            f"signature length mismatch: {len(sig_a)} vs {len(sig_b)}"
        )
    if not sig_a:
        return 0.0
    matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return matches / len(sig_a)
