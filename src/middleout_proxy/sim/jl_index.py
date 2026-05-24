"""Hybrid JL + MinHash LSH index.

Drop-in replacement for :class:`middleout_proxy.jl.RequestSketchIndex`. Uses a
banded MinHash LSH to pick candidate records cheaply, then ranks each candidate
by JL signed-projection cosine. Falls back to a full JL scan when the corpus is
small (``< 16`` records) so that tiny in-request scans behave identically to
the simple ``RequestSketchIndex``.
"""

from __future__ import annotations

from ..jl import SketchRecord, cosine, signed_jl_projection
from .lsh import MinHashLSH
from .minhash import minhash_signature


class HybridSketchIndex:
    """JL + LSH dedupe index keyed by JL cosine.

    The public surface mirrors :class:`middleout_proxy.jl.RequestSketchIndex`
    so callers can swap one for the other without touching consumer code.
    """

    def __init__(
        self,
        *,
        jl_dims: int = 512,
        jl_shingle_tokens: int = 5,
        mh_num_perms: int = 128,
        mh_bands: int = 32,
        mh_shingle_size: int = 5,
        small_corpus_cutoff: int = 16,
    ) -> None:
        self.jl_dims = jl_dims
        self.jl_shingle_tokens = jl_shingle_tokens
        self.mh_num_perms = mh_num_perms
        self.mh_bands = mh_bands
        self.mh_shingle_size = mh_shingle_size
        self.small_corpus_cutoff = small_corpus_cutoff
        self._records: list[SketchRecord] = []
        self._record_by_id: dict[str, SketchRecord] = {}
        self._lsh = MinHashLSH(num_perms=mh_num_perms, bands=mh_bands)

    def add(self, *, text: str, path: str, digest: str) -> None:
        sketch = signed_jl_projection(
            text, dims=self.jl_dims, shingle_tokens=self.jl_shingle_tokens
        )
        record = SketchRecord(path=path, digest=digest, chars=len(text), sketch=sketch)
        doc_id = self._make_doc_id(path, digest, len(self._records))
        self._records.append(record)
        self._record_by_id[doc_id] = record

        sig = minhash_signature(
            text, num_perms=self.mh_num_perms, shingle_size=self.mh_shingle_size
        )
        self._lsh.add(doc_id, sig)

    def find_best(self, text: str) -> tuple[SketchRecord | None, float]:
        if not self._records:
            return None, -1.0

        sketch = signed_jl_projection(
            text, dims=self.jl_dims, shingle_tokens=self.jl_shingle_tokens
        )

        # Tiny corpora: a brute-force JL scan is cheaper than building a MinHash
        # and walking the LSH buckets, and avoids the LSH's recall floor.
        if len(self._records) < self.small_corpus_cutoff:
            return self._scan(sketch, self._records)

        sig = minhash_signature(
            text,
            num_perms=self.mh_num_perms,
            shingle_size=self.mh_shingle_size,
        )
        candidate_ids = self._lsh.candidates(sig)
        if not candidate_ids:
            return None, -1.0

        candidates = [
            self._record_by_id[c] for c in candidate_ids if c in self._record_by_id
        ]
        if not candidates:
            return None, -1.0
        return self._scan(sketch, candidates)

    def __len__(self) -> int:
        return len(self._records)

    @staticmethod
    def _make_doc_id(path: str, digest: str, index: int) -> str:
        return f"{path}#{digest}#{index}"

    @staticmethod
    def _scan(
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
