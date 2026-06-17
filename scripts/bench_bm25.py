"""Benchmark the pure-Python BM25 retriever — a baseline to compare a future
parallel Rust implementation against.

Measures the two costs that matter:
  1. BUILD  — BM25Retriever.add() over the whole corpus (tokenize + index).
  2. QUERY  — retrieve() latency (mean / p50 / p95 / max) and throughput.

Loads chunks straight from the saved index's chunks.jsonl, so it touches BM25
only — no embedding model, no FAISS — and the numbers are clean.

Run: PYTHONPATH=src uv run python scripts/bench_bm25.py
"""
from __future__ import annotations

import time
from pathlib import Path

from scholar_rag.models import Chunk
from scholar_rag.retrieval.bm25 import BM25Retriever, tokenize

CHUNKS_FILE = Path("data/index/chunks.jsonl")
K = 10
REPEATS = 5  # times each query is run, for stable latency stats

QUERIES = [
    "reconstruction number of graphs",
    "chromatic number and clique number",
    "split graphs edge reconstruction",
    "spectral properties of trees",
    "domination number bounds",
    "planar graph coloring algorithm",
    "random graph phase transition",
    "graph isomorphism complexity",
    "hamiltonian cycle existence conditions",
    "eigenvalues of the adjacency matrix",
    "independent set approximation",
    "treewidth and dynamic programming",
]


def percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round(pct / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def main() -> None:
    chunks = [Chunk.model_validate_json(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]
    total_tokens = sum(len(tokenize(c.text)) for c in chunks)

    # --- BUILD ---
    bm25 = BM25Retriever()
    t0 = time.perf_counter()
    bm25.add(chunks)
    build_s = time.perf_counter() - t0

    # --- QUERY ---
    for q in QUERIES:  # warmup (caches, branch prediction, etc.)
        bm25.retrieve(q, k=K)

    samples_ms: list[float] = []
    for _ in range(REPEATS):
        for q in QUERIES:
            t = time.perf_counter()
            bm25.retrieve(q, k=K)
            samples_ms.append((time.perf_counter() - t) * 1000)
    samples_ms.sort()

    mean_ms = sum(samples_ms) / len(samples_ms)

    print("=" * 56)
    print("BM25 BENCHMARK — pure-Python baseline")
    print("=" * 56)
    print(f"corpus      : {len(chunks):,} chunks, {len(bm25._df):,} vocab terms")
    print(f"total tokens: {total_tokens:,}  (avgdl {total_tokens/len(chunks):.0f})")
    print("-" * 56)
    print(f"BUILD       : {build_s:.3f} s   ({len(chunks)/build_s:,.0f} chunks/s)")
    print("-" * 56)
    print(f"QUERY (k={K}, {len(QUERIES)} queries x {REPEATS} = {len(samples_ms)} samples)")
    print(f"  mean      : {mean_ms:.2f} ms")
    print(f"  p50       : {percentile(samples_ms, 50):.2f} ms")
    print(f"  p95       : {percentile(samples_ms, 95):.2f} ms")
    print(f"  max       : {samples_ms[-1]:.2f} ms")
    print(f"  throughput: {1000/mean_ms:,.0f} queries/s  (single-threaded)")
    print("=" * 56)


if __name__ == "__main__":
    main()
