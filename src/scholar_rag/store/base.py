from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from scholar_rag.models import Chunk, RetrievedChunk

class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        """Embed and store chunks. Idempotent on chunk_id"""

    @abstractmethod
    def search(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        """Return the k most similar chunks to `query`, highest score first"""

    @abstractmethod
    def __len__(self) -> int:
        """Number of chunks stored"""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the store to a directory"""