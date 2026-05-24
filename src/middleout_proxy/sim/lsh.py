"""Banded LSH index over MinHash signatures.

A signature of length ``num_perms`` is split into ``bands`` slices of
``rows = num_perms / bands`` ints each. Two documents are *candidates* if any
single band hashes to the same bucket — i.e. if any of their band slices are
byte-identical. The probability of being candidates given Jaccard ``s`` is

    P(candidate) ≈ 1 - (1 - s ** rows) ** bands

so tuning ``(bands, rows)`` trades recall against false positives. With
``num_perms=128, bands=32`` (rows=4) the S-curve crosses 0.5 around ``s ≈ 0.5``
which is well above the noise floor for "near duplicate" use.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field


@dataclass
class MinHashLSH:
    """In-memory banded LSH index.

    ``add(doc_id, signature)``        — insert a doc.
    ``candidates(signature)``         — set of doc ids sharing at least one band.
    ``remove(doc_id)``                — pull a doc out of every band it landed in.
    ``len(index)``                    — number of inserted docs.
    """

    num_perms: int = 128
    bands: int = 32
    rows: int = field(init=False)
    _buckets: list[dict[bytes, list[str]]] = field(init=False, repr=False)
    _band_keys_by_doc: dict[str, list[bytes]] = field(init=False, repr=False)

    def __init__(self, *, num_perms: int = 128, bands: int = 32) -> None:
        if num_perms <= 0:
            raise ValueError("num_perms must be positive")
        if bands <= 0:
            raise ValueError("bands must be positive")
        if num_perms % bands != 0:
            raise ValueError(
                f"bands ({bands}) must divide num_perms ({num_perms}) evenly"
            )
        self.num_perms = num_perms
        self.bands = bands
        self.rows = num_perms // bands
        self._buckets = [dict() for _ in range(bands)]
        self._band_keys_by_doc = {}

    # ----- public API -----

    def add(self, doc_id: str, signature: tuple[int, ...]) -> None:
        self._validate_sig(signature)
        if doc_id in self._band_keys_by_doc:
            # Idempotent re-add: drop the old entry first so we don't double-list.
            self.remove(doc_id)
        keys = self._band_keys(signature)
        for band_idx, key in enumerate(keys):
            bucket = self._buckets[band_idx].setdefault(key, [])
            bucket.append(doc_id)
        self._band_keys_by_doc[doc_id] = keys

    def candidates(self, signature: tuple[int, ...]) -> set[str]:
        self._validate_sig(signature)
        keys = self._band_keys(signature)
        out: set[str] = set()
        for band_idx, key in enumerate(keys):
            bucket = self._buckets[band_idx].get(key)
            if bucket:
                out.update(bucket)
        return out

    def remove(self, doc_id: str) -> None:
        keys = self._band_keys_by_doc.pop(doc_id, None)
        if keys is None:
            return
        for band_idx, key in enumerate(keys):
            bucket = self._buckets[band_idx].get(key)
            if not bucket:
                continue
            try:
                bucket.remove(doc_id)
            except ValueError:
                pass
            if not bucket:
                self._buckets[band_idx].pop(key, None)

    def __len__(self) -> int:
        return len(self._band_keys_by_doc)

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._band_keys_by_doc

    # ----- internals -----

    def _validate_sig(self, signature: tuple[int, ...]) -> None:
        if len(signature) != self.num_perms:
            raise ValueError(
                f"signature length {len(signature)} != num_perms {self.num_perms}"
            )

    def _band_keys(self, signature: tuple[int, ...]) -> list[bytes]:
        rows = self.rows
        pack_fmt = "!" + "Q" * rows
        keys: list[bytes] = []
        for band_idx in range(self.bands):
            start = band_idx * rows
            slab = struct.pack(pack_fmt, *signature[start : start + rows])
            keys.append(
                hashlib.blake2b(slab, digest_size=16).digest()
            )
        return keys
