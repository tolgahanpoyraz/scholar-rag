from __future__ import annotations

import json
from pathlib import Path

from scholar_rag.embedding.base import Embedder
from scholar_rag.models import Chunk, RetrievedChunk
from scholar_rag.store.base import VectorStore

_INDEX_FILE = "index.faiss"
_CHUNKS_FILE = "chunks.jsonl"

class FaissVectorStore(VectorStore):
    def __init__(self, embedder: Embedder):
        import faiss

        self._embedder = embedder
        self._index = faiss.IndexFlatIP(embedder.dimension)
        self._chunks: list[Chunk] = []
        self._seen: set[str] = set()

    def add(self, chunks: list[Chunk]) -> None:
        import numpy as np

        new = [c for c in chunks if c.chunk_id not in self._seen]
        if not new:
            return
        vectors = self._embedder.embed_texts([c.text for c in new])
        arr = np.array(vectors, dtype="float32")
        self._index.add(arr)
        self._chunks.extend(new)
        self._seen.update(c.chunk_id for c in new)

    def search(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        import numpy as np

        if not self._chunks:
            return []

        qv = self._embedder.embed_query(query)
        arr = np.array([qv], dtype="float32")
        k = min(k, len(self._chunks))
        scores, idxs = self._index.search(arr, k)
        results: list[RetrievedChunk] = []

        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append(
                RetrievedChunk(
                    chunk=self._chunks[idx],
                    score=float(score),
                    retrieval_source="dense",
                )
            )
        return results

    def __len__(self) -> int:
        return len(self._chunks)

    def save(self, path: Path) -> None:
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / _INDEX_FILE))

        with open(path / _CHUNKS_FILE, "w") as f:
            for c in self._chunks:
                f.write(c.model_dump_json() + "\n")

    @classmethod
    def load(cls, path: Path, embedder: Embedder) -> "FaissVectorStore":
        import faiss

        path = Path(path)
        store = cls(embedder)
        store._index = faiss.read_index(str(path / _INDEX_FILE))
        with open(path / _CHUNKS_FILE) as f:
            store._chunks = [Chunk.model_validate_json(line) for line in f if line.strip()]
        store._seen = {c.chunk_id for c in store._chunks}
        return store

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)