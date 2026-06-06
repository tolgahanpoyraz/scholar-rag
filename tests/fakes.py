from __future__ import annotations

import hashlib
import math

from scholar_rag.embedding.base import Embedder
from scholar_rag.generation.base import Generator
from scholar_rag.models import Answer, RetrievedChunk

class FakeEmbedder(Embedder):
    def __init__(self, dimension: int = 8) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            vec = [h[i % len(h)] / 255.0 for i in range(self._dim)]
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out

class FakeGenerator(Generator):
    def generate(self, query: str, chunks: list[RetrievedChunk]) -> Answer:
        cited = ", ".join(f"[{i}]" for i in range(1, len(chunks) + 1)) or "(none)"
        return Answer(text=f"Answer to '{query}' grounded in {cited}.", citations=chunks)