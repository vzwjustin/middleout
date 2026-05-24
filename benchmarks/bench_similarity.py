"""Compare JL / MinHash / SimHash / Hybrid on synthetic near-duplicate text.

Run::

    python -m benchmarks.bench_similarity
    # or
    python benchmarks/bench_similarity.py

Environment overrides (all integers):

* ``BENCH_DOCS``    — corpus size (default 60).
* ``BENCH_DOC_SIZE`` — target chars per doc (default 1024).
* ``BENCH_QUERIES``  — number of query docs (default 12).

For each overlap level (``0%``, ``50%``, ``99%``) the bench generates a corpus,
builds four indexes (JL brute-force, MinHash brute-force, SimHash brute-force,
JL+LSH hybrid), times them, and reports precision / recall@1 against an exact
word-shingle Jaccard ground truth at threshold 0.85. Output is a single
Markdown table on stdout — no external libraries.
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

# Self-bootstrap so the script works without an install or PYTHONPATH.
_THIS = Path(__file__).resolve()
_SRC = _THIS.parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from middleout_proxy.jl import RequestSketchIndex, tokenize_words  # noqa: E402
from middleout_proxy.sim.jl_index import HybridSketchIndex  # noqa: E402
from middleout_proxy.sim.minhash import jaccard_estimate, minhash_signature  # noqa: E402
from middleout_proxy.sim.simhash import simhash64, simhash_similarity  # noqa: E402

THRESHOLD = 0.85
SHINGLE = 5


# ----- synthetic corpus -----

def _vocab(seed: int, size: int = 600) -> list[str]:
    rng = random.Random(seed)
    return [f"w{rng.randint(0, 9999):04d}" for _ in range(size)]


def _make_doc(rng: random.Random, vocab: list[str], target_chars: int) -> str:
    words: list[str] = []
    chars = 0
    while chars < target_chars:
        w = rng.choice(vocab)
        words.append(w)
        chars += len(w) + 1
    return " ".join(words)


def _variant(doc: str, overlap: float, rng: random.Random) -> str:
    """Produce a variant of ``doc`` that shares ~``overlap`` fraction of words."""
    if overlap >= 0.999:
        return doc
    words = doc.split()
    if overlap <= 0.001:
        # Fully unrelated text — keep length similar so length isn't a signal.
        return " ".join(f"q{rng.randint(0, 9999):04d}" for _ in words)
    out = []
    for w in words:
        if rng.random() < overlap:
            out.append(w)
        else:
            out.append(f"alt{rng.randint(0, 99999):05d}")
    return " ".join(out)


# ----- exact ground truth via word k-shingle Jaccard -----

def _shingle_set(text: str) -> set[str]:
    toks = tokenize_words(text)
    if not toks:
        return set()
    if len(toks) <= SHINGLE:
        return {" ".join(toks)}
    return {" ".join(toks[i : i + SHINGLE]) for i in range(len(toks) - SHINGLE + 1)}


def _exact_jaccard(a_shingles: set[str], b_shingles: set[str]) -> float:
    if not a_shingles or not b_shingles:
        return 0.0
    inter = len(a_shingles & b_shingles)
    union = len(a_shingles | b_shingles)
    return inter / union if union else 0.0


def _ground_truth(corpus: list[str], queries: list[str]) -> list[tuple[int, float]]:
    corpus_shingles = [_shingle_set(d) for d in corpus]
    out: list[tuple[int, float]] = []
    for q in queries:
        qs = _shingle_set(q)
        best_i, best_j = -1, -1.0
        for i, cs in enumerate(corpus_shingles):
            j = _exact_jaccard(qs, cs)
            if j > best_j:
                best_j = j
                best_i = i
        out.append((best_i, best_j))
    return out


# ----- index runners -----

def _run_jl(corpus: list[str], queries: list[str]) -> dict[str, object]:
    idx = RequestSketchIndex(dims=512, shingle_tokens=SHINGLE)
    t0 = time.perf_counter()
    for i, d in enumerate(corpus):
        idx.add(text=d, path=f"d#{i}", digest=str(i))
    build_ms = (time.perf_counter() - t0) * 1000.0

    preds: list[tuple[int, float]] = []
    t0 = time.perf_counter()
    for q in queries:
        rec, score = idx.find_best(q)
        pred_i = -1 if rec is None else int(rec.digest)
        preds.append((pred_i, max(0.0, min(1.0, score))))
    query_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(queries))
    return {"build_ms": build_ms, "query_ms": query_ms, "preds": preds}


def _run_hybrid(corpus: list[str], queries: list[str]) -> dict[str, object]:
    idx = HybridSketchIndex(jl_dims=512, jl_shingle_tokens=SHINGLE)
    t0 = time.perf_counter()
    for i, d in enumerate(corpus):
        idx.add(text=d, path=f"d#{i}", digest=str(i))
    build_ms = (time.perf_counter() - t0) * 1000.0

    preds: list[tuple[int, float]] = []
    t0 = time.perf_counter()
    for q in queries:
        rec, score = idx.find_best(q)
        pred_i = -1 if rec is None else int(rec.digest)
        preds.append((pred_i, max(0.0, min(1.0, score))))
    query_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(queries))
    return {"build_ms": build_ms, "query_ms": query_ms, "preds": preds}


def _run_minhash(corpus: list[str], queries: list[str]) -> dict[str, object]:
    t0 = time.perf_counter()
    sigs = [minhash_signature(d) for d in corpus]
    build_ms = (time.perf_counter() - t0) * 1000.0

    preds: list[tuple[int, float]] = []
    t0 = time.perf_counter()
    for q in queries:
        qs = minhash_signature(q)
        best_i, best_j = -1, -1.0
        for i, s in enumerate(sigs):
            j = jaccard_estimate(qs, s)
            if j > best_j:
                best_j = j
                best_i = i
        preds.append((best_i, max(0.0, min(1.0, best_j))))
    query_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(queries))
    return {"build_ms": build_ms, "query_ms": query_ms, "preds": preds}


def _run_simhash(corpus: list[str], queries: list[str]) -> dict[str, object]:
    t0 = time.perf_counter()
    hashes = [simhash64(d) for d in corpus]
    build_ms = (time.perf_counter() - t0) * 1000.0

    preds: list[tuple[int, float]] = []
    t0 = time.perf_counter()
    for q in queries:
        qh = simhash64(q)
        best_i, best_s = -1, -1.0
        for i, h in enumerate(hashes):
            s = simhash_similarity(qh, h)
            if s > best_s:
                best_s = s
                best_i = i
        preds.append((best_i, max(0.0, min(1.0, best_s))))
    query_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(queries))
    return {"build_ms": build_ms, "query_ms": query_ms, "preds": preds}


# ----- scoring -----

def _score(
    preds: list[tuple[int, float]], truth: list[tuple[int, float]]
) -> tuple[float, float]:
    """Return (precision, recall@1) at THRESHOLD."""
    tp = fp = fn = 0
    for (pi, ps), (ti, tj) in zip(preds, truth):
        truth_pos = tj >= THRESHOLD
        pred_pos = ps >= THRESHOLD
        if truth_pos and pred_pos:
            if pi == ti:
                tp += 1
            else:
                fp += 1
        elif (not truth_pos) and pred_pos:
            fp += 1
        elif truth_pos and (not pred_pos):
            fn += 1
        # else TN, ignored
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return precision, recall


# ----- top-level -----

def main() -> int:
    docs = int(os.getenv("BENCH_DOCS", "60"))
    doc_size = int(os.getenv("BENCH_DOC_SIZE", "1024"))
    queries_n = int(os.getenv("BENCH_QUERIES", "12"))

    overlaps = [("0%", 0.0), ("50%", 0.5), ("99%", 0.99)]
    algorithms = (
        ("JL", _run_jl),
        ("MinHash", _run_minhash),
        ("SimHash", _run_simhash),
        ("Hybrid", _run_hybrid),
    )

    print(
        f"# Similarity bench (docs={docs}, doc_size={doc_size}, queries={queries_n}, "
        f"threshold={THRESHOLD})"
    )
    print()
    print("| Algorithm | Overlap | Build (ms) | Query (ms) | Precision | Recall@1 |")
    print("| --------- | ------- | ---------- | ---------- | --------- | -------- |")

    for label, overlap in overlaps:
        rng = random.Random(0xC0FFEE ^ hash(label) & 0xFFFF)
        vocab = _vocab(seed=42)
        corpus = [_make_doc(rng, vocab, doc_size) for _ in range(docs)]
        query_idxs = rng.sample(range(docs), min(queries_n, docs))
        queries = [_variant(corpus[i], overlap, rng) for i in query_idxs]
        truth = _ground_truth(corpus, queries)
        # Truth ids are positions inside corpus. Replace the position with the
        # actual query-source index for cleaner comparison: high-overlap queries
        # should ground-truth-back to their source doc, but exact Jaccard top-1
        # can occasionally tie elsewhere — we trust the exact computation.

        for name, runner in algorithms:
            result = runner(corpus, queries)
            preds: list[tuple[int, float]] = result["preds"]  # type: ignore[assignment]
            precision, recall = _score(preds, truth)
            print(
                f"| {name:<9} | {label:<7} | {result['build_ms']:>10.2f} "
                f"| {result['query_ms']:>10.3f} | {precision:>9.3f} | {recall:>8.3f} |"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
