from __future__ import annotations
import math
import re
from collections import Counter

from scholar_rag.models import Chunk, RetrievedChunk
from scholar_rag.retrieval.base import Retriever

_TOKEN_RE = re.compile(r"\w+")

_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with from into over
    under again further is are was were be been being am do does did doing have
    has had having this that these those it its as not no nor so than too very
    can will just we you your they them their he she his her i me my our us
    """.split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]

class BM25Retriever(Retriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._chunks: list[Chunk] = []
        self._doc_tokens: list[list[str]] = []
        self._tf: list[Counter] = []
        self._df: dict[str, int] = {}
        self.avgdl = 0.0
        self._seen: set[str] = set()

    def add(self, chunks: list[Chunk]) -> None:
        new = [c for c in chunks if c.chunk_id not in self._seen]
        for c in new:
            toks = tokenize(c.text)
            self._chunks.append(c)
            self._doc_tokens.append(toks)
            tf = Counter(toks)
            self._tf.append(tf)
            for term in tf:
                self._df[term] = self._df.get(term, 0) + 1
            self._seen.add(c.chunk_id)
        if self._doc_tokens:
            self._avgdl = sum(len(t) for t in self._doc_tokens) / len(self._doc_tokens)


    def _idf(self, term: str, n_docs: int) -> float:
        df = self._df.get(term, 0)
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def retrieve(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        q_terms = tokenize(query)
        n = len(self._chunks)
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            dl = len(self._doc_tokens[i])
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf(term, n)
                denom = f + self._k1 * (1 - self._b + self._b * dl / self._avgdl)
                score += idf * (f * (self._k1 + 1)) / denom
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [
            RetrievedChunk(chunk=self._chunks[i], score=float(s), retrieval_source="bm25")
            for s, i in scored[:k]
        ]
