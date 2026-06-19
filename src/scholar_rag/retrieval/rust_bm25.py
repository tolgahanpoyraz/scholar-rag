"""BM25 retriever whose per-query scoring loop runs in Rust.

Same interface and same math as the pure-Python `BM25Retriever`; the difference
is purely where the hot loop executes. Python does the tokenizing,
the vocabulary, document frequencies, top-k selection; Rust does the scan over
all documents

The handoff is plain data: words become integer ids, and the
per-document term frequencies are flattened into three parallel arrays that Rust can store once and scan fast.
"""
from __future__ import annotations

import math
from collections import Counter

import scholar_rag_bm25 as _rs

from scholar_rag.models import Chunk, RetrievedChunk
from scholar_rag.retrieval.base import Retriever
from scholar_rag.retrieval.bm25 import tokenize


class RustBM25Retriever(Retriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._chunks: list[Chunk] = []
        self._vocab: dict[str, int] = {}     # word -> integer id
        self._df: dict[int, int] = {}        # term id -> # docs containing it
        self._tfs: list[Counter] = []        # per doc: term id -> freq
        self._seen: set[str] = set()
        self._index: _rs.Bm25Index | None = None

    def _term_id(self, term: str) -> int:
        tid = self._vocab.get(term)
        if tid is None:
            tid = len(self._vocab)
            self._vocab[term] = tid
        return tid

    def add(self, chunks: list[Chunk]) -> None:
        new = [c for c in chunks if c.chunk_id not in self._seen]
        for c in new:
            ids = [self._term_id(t) for t in tokenize(c.text)]
            tf = Counter(ids)
            self._chunks.append(c)
            self._tfs.append(tf)
            for tid in tf:
                self._df[tid] = self._df.get(tid, 0) + 1
            self._seen.add(c.chunk_id)
        if self._chunks:
            self._build_index()

    def _build_index(self) -> None:
        """Flatten the per-doc term frequencies into the CSR arrays Rust holds."""
        offsets = [0]
        term_ids: list[int] = []
        freqs: list[int] = []
        lens: list[float] = []
        for tf in self._tfs:
            for tid, f in tf.items():
                term_ids.append(tid)
                freqs.append(f)
            offsets.append(len(term_ids))
            lens.append(float(sum(tf.values())))
        self._avgdl = (sum(lens) / len(lens)) if lens else 0.0
        self._index = _rs.Bm25Index(
            offsets, term_ids, freqs, lens, self._avgdl, self._k1, self._b
        )

    def _idf(self, tid: int, n_docs: int) -> float:
        df = self._df.get(tid, 0)
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def retrieve(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        if self._index is None or not self._chunks:
            return []
        n = len(self._chunks)

        # Map query words to ids, dropping any the corpus never saw (they'd score
        # 0 anyway). A repeated query word counts multiple times in the reference
        # implementation, so fold that count into the idf to match it exactly.
        q_ids = [self._vocab[t] for t in tokenize(query) if t in self._vocab]
        if not q_ids:
            return []
        counts = Counter(q_ids)
        term_ids = list(counts.keys())
        idfs = [self._idf(tid, n) * cnt for tid, cnt in counts.items()]

        # Rust does the scan AND the top-k selection; we just get k winners back.
        ranked = self._index.top_k(term_ids, idfs, k)
        return [
            RetrievedChunk(chunk=self._chunks[i], score=float(s), retrieval_source="bm25")
            for s, i in ranked
        ]


class InvertedRustBM25Retriever(RustBM25Retriever):
    def _build_index(self) -> None:
        vocab_size = len(self._vocab)
        bucket_docs: list[list[int]] = [[] for _ in range(vocab_size)]
        bucket_freqs: list[list[int]] = [[] for _ in range(vocab_size)]
        lens: list[float] = []
        for doc_idx, tf in enumerate(self._tfs):
            lens.append(float(sum(tf.values())))
            for tid, f in tf.items():
                bucket_docs[tid].append(doc_idx)
                bucket_freqs[tid].append(f)

        term_offsets = [0]
        posting_docs: list[int] = []
        posting_freqs: list[int] = []
        for tid in range(vocab_size):
            posting_docs.extend(bucket_docs[tid])
            posting_freqs.extend(bucket_freqs[tid])
            term_offsets.append(len(posting_docs))

        self._avgdl = (sum(lens) / len(lens)) if lens else 0.0
        self._index = _rs.Bm25Inverted(
            term_offsets, posting_docs, posting_freqs, lens, self._avgdl, self._k1, self._b
        )