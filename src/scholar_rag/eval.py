from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from scholar_rag.retrieval.base import Retriever


@dataclass(frozen=True)
class EvalItem:
    query: str
    relevant_docs: list[str]


@dataclass(frozen=True)
class EvalResult:
    recall_at_k: float
    mrr: float
    n: int
    k: int

    def __str__(self) -> str:
        return f"n={self.n}  recall@{self.k}={self.recall_at_k:.3f}  MRR={self.mrr:.3f}"


def load_dataset(path: Path) -> list[EvalItem]:
    data = json.loads(Path(path).read_text())
    return [EvalItem(query=d["query"], relevant_docs=d["relevant_docs"]) for d in data]


def recall_at_k(retrieved_doc_ids: list[str], relevant: list[str], k: int) -> float:
    return 1.0 if set(relevant) & set(retrieved_doc_ids[:k]) else 0.0


def reciprocal_rank(retrieved_doc_ids: list[str], relevant: list[str]) -> float:
    rel = set(relevant)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in rel:
            return 1.0 / rank
    return 0.0


def evaluate(retriever: Retriever, dataset: list[EvalItem], k: int = 10) -> EvalResult:
    recalls: list[float] = []
    rrs: list[float] = []
    for item in dataset:
        results = retriever.retrieve(item.query, k)
        doc_ids = [r.chunk.source_doc_id for r in results]
        recalls.append(recall_at_k(doc_ids, item.relevant_docs, k))
        rrs.append(reciprocal_rank(doc_ids, item.relevant_docs))
    return EvalResult(
        recall_at_k=mean(recalls) if recalls else 0.0,
        mrr=mean(rrs) if rrs else 0.0,
        n=len(dataset),
        k=k,
    )
