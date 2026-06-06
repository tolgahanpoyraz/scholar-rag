from __future__ import annotations

from abc import ABC, abstractmethod

from scholar_rag.models import Answer, RetrievedChunk

class Generator(ABC):
    @abstractmethod
    def generate(self, query: str, chunks: list[RetrievedChunk]) -> Answer:
        """Synthesize an answer to `query` grounded in `chunks`"""