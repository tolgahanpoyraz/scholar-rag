from __future__ import annotations

from scholar_rag.models import RetrievedChunk
from scholar_rag.retrieval.base import Retriever
from scholar_rag.store.base import VectorStore

class DenseRetriever(Retriever):
    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def retrieve(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        return self._store.search(query, k)