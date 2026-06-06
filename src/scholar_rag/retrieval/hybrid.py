from __future__ import annotations

from scholar_rag.models import RetrievedChunk
from scholar_rag.retrieval.base import Retriever

class HybridRetriever(Retriever):
    def __init__(
            self,
            retrievers: list[Retriever],
            rrf_k: int = 60,
            fetch_k: int = 20,
        ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self._retrievers = retrievers
        self._rrf_k = rrf_k
        self._fetch_k = fetch_k

    def retrieve(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        fused: dict[str, float] = {}
        by_id: dict[str, RetrievedChunk] = {}
        for retriever in self._retrievers:
            results = retriever.retrieve(query, self._fetch_k)
            for rank, rc in enumerate(results, start=1):
                cid = rc.chunk.chunk_id
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self._rrf_k + rank)
                by_id.setdefault(cid, rc)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        return [
            RetrievedChunk(chunk=by_id[cid].chunk, score=float(score),
                           retrieval_source="fused")
            for cid, score in ranked[:k]
        ]