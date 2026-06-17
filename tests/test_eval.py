from __future__ import annotations

from scholar_rag.eval import (
    EvalItem,
    evaluate,
    recall_at_k,
    reciprocal_rank,
)
from scholar_rag.models import Chunk, RetrievedChunk
from scholar_rag.retrieval.base import Retriever


def test_recall_at_k():
    assert recall_at_k(["A", "B", "C"], ["A"], 3) == 1.0
    assert recall_at_k(["X", "Y", "A"], ["A"], 2) == 0.0
    assert recall_at_k(["X", "Y", "A"], ["A"], 3) == 1.0
    assert recall_at_k(["X", "Y", "Z"], ["A"], 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["A", "B"], ["A"]) == 1.0
    assert reciprocal_rank(["X", "Y", "A"], ["A"]) == 1.0 / 3
    assert reciprocal_rank(["X"], ["A"]) == 0.0


class _ScriptedRetriever(Retriever):
    def __init__(self, by_query: dict[str, list[str]]):
        self._by_query = by_query

    def retrieve(self, query: str, k: int = 10):
        doc_ids = self._by_query.get(query, [])[:k]
        return [
            RetrievedChunk(
                chunk=Chunk(source="t", source_doc_id=d, chunk_index=i, text="x"),
                score=1.0,
            )
            for i, d in enumerate(doc_ids)
        ]


def test_evaluate_aggregates():
    retriever = _ScriptedRetriever({
        "q1": ["docA", "docB"],
        "q2": ["docX", "docY"],
    })
    dataset = [
        EvalItem("q1", ["docA"]),
        EvalItem("q2", ["docZ"]),
    ]
    result = evaluate(retriever, dataset, k=5)
    assert result.n == 2
    assert result.recall_at_k == 0.5
    assert result.mrr == 0.5