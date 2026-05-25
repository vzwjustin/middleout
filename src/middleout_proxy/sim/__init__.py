"""Similarity primitives for middleout-proxy.

Public API is re-exported here so callers can do::

    from middleout_proxy.sim import (
        minhash_signature,
        jaccard_estimate,
        simhash64,
        hamming_distance,
        simhash_similarity,
        MinHashLSH,
        HybridSketchIndex,
    )

The underlying modules (``minhash``, ``simhash``, ``lsh``, ``jl_index``) are
also importable directly if you need the implementation surface.
"""

from __future__ import annotations

from .jl_index import HybridSketchIndex
from .lsh import MinHashLSH
from .minhash import jaccard_estimate, minhash_signature
from .simhash import hamming_distance, simhash64, simhash_similarity

__all__ = [
    "HybridSketchIndex",
    "MinHashLSH",
    "hamming_distance",
    "jaccard_estimate",
    "minhash_signature",
    "simhash64",
    "simhash_similarity",
]
