from __future__ import annotations

from scholar_rag.models import Chunk, RetrievedChunk
from scholar_rag.retrieval.base import Retriever
from scholar_rag.retrieval.bm25 import BM25Retriever, tokenize
from scholar_rag.retrieval.hybrid import HybridRetriever


def _chunk(idx: int, text: str) -> Chunk:
    return Chunk(source="t", source_doc_id="d1", chunk_index=idx, text=text)


def test_tokenize_lowercases_and_splits():
    assert tokenize("ERN(G) is split") == ["ern", "g", "split"]


def test_tokenize_drops_stopwords():
    assert tokenize("the spectral properties of a graph") == ["spectral", "properties", "graph"]


def test_bm25_ranks_exact_term_first():
    bm = BM25Retriever()
    bm.add([
        _chunk(0, "the degree associated edge reconstruction number dern of split graphs"),
        _chunk(1, "the degree associated reconstruction number drn uses vertex cards"),
        _chunk(2, "a split graph is a clique plus an independent set"),
    ])
    top = bm.retrieve("dern", k=1)
    assert top[0].chunk.chunk_index == 0
    assert top[0].retrieval_source == "bm25"


def test_bm25_empty_returns_empty():
    assert BM25Retriever().retrieve("anything") == []


def test_bm25_idempotent_add():
    bm = BM25Retriever()
    bm.add([_chunk(0, "split graph clique")])
    bm.add([_chunk(0, "split graph clique")])
    assert len(bm._chunks) == 1


class _FakeRetriever(Retriever):
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
    def retrieve(self, query: str, k: int = 10):
        return [RetrievedChunk(chunk=c, score=1.0, retrieval_source="fake")
                for c in self._chunks[:k]]


def test_rrf_rewards_agreement_across_retrievers():
    a, b, c = _chunk(0, "A"), _chunk(1, "B"), _chunk(2, "C")
    dense = _FakeRetriever([a, b, c])
    bm25 = _FakeRetriever([c, a, b])
    hybrid = HybridRetriever([dense, bm25])
    results = hybrid.retrieve("q", k=3)
    assert results[0].chunk.chunk_index == 0
    assert results[0].retrieval_source == "fused"


def test_hybrid_requires_a_retriever():
    import pytest
    with pytest.raises(ValueError):
        HybridRetriever([])