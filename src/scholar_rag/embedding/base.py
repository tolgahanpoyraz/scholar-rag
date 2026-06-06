from __future__ import annotations

from abc import ABC, abstractmethod

class Embedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension. The store needs this to allocate its index"""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents"""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query"""
        return self.embed_texts([text])[0]