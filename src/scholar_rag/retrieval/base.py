from __future__ import annotations

from abc import ABC, abstractmethod

from scholar_rag.models import RetrievedChunk

class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        """Return the k best chunks for 'query', highest-ranekd first."""
