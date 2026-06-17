"""The RAG pipeline, assembled once and reused for every request.

This is the same wiring as scripts/ingest_and_query.py — load the index, build
the retrievers, create the generator — but packaged as an object so the web app
can build it a single time at startup (loading the embedding model and FAISS
index is slow) and then answer many questions cheaply.

Nothing in this file knows about HTTP. It speaks only in domain terms: you give
it a question string, it gives you back an `Answer` (see scholar_rag/models.py).
"""
from __future__ import annotations

from collections.abc import Iterator

from scholar_rag.config import Config, load_config
from scholar_rag.embedding.sentence_transformer import SentenceTransformerEmbedder
from scholar_rag.generation.base import Generator
from scholar_rag.generation.openrouter import OpenRouterGenerator
from scholar_rag.models import Answer, RetrievedChunk
from scholar_rag.retrieval.base import Retriever
from scholar_rag.retrieval.bm25 import BM25Retriever
from scholar_rag.retrieval.dense import DenseRetriever
from scholar_rag.retrieval.hybrid import HybridRetriever
from scholar_rag.store.faiss_store import FaissVectorStore


class RagEngine:
    """A ready-to-query RAG pipeline: retriever + generator + settings."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        top_k: int,
        library_size: int,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k
        self._library_size = library_size

    @classmethod
    def from_config(cls, config: Config | None = None) -> "RagEngine":
        """Assemble the pipeline from a saved index on disk.

        Expensive: loads the embedding model and reads the FAISS index. Call
        this once (at app startup), not per request.
        """
        config = config or load_config()
        index_dir = config.data_dir / "index"

        embedder = SentenceTransformerEmbedder(config.embedding_model)
        store = FaissVectorStore.load(index_dir, embedder)

        bm25 = BM25Retriever()
        bm25.add(store.all_chunks())
        retriever = HybridRetriever([DenseRetriever(store), bm25])

        # Distinct source documents — what the UI header calls "papers in library".
        library_size = len({c.source_doc_id for c in store.all_chunks()})

        return cls(
            retriever=retriever,
            generator=OpenRouterGenerator(),
            top_k=config.top_k,
            library_size=library_size,
        )

    @property
    def library_size(self) -> int:
        return self._library_size

    def ask(self, question: str) -> Answer:
        """Retrieve the best passages for `question`, then generate a cited answer.

        If retrieval finds nothing, the generator is told there are no passages
        and is prompted to say it can't answer — we never fabricate sources.
        """
        return self._generator.generate(question, self.retrieve(question))

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """The retrieval half of `ask`, exposed so the streaming endpoint can
        send the sources to the client before generation begins."""
        return self._retriever.retrieve(question, k=self._top_k)

    def stream(self, question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
        """The generation half: yield answer text pieces for already-retrieved chunks."""
        return self._generator.generate_stream(question, chunks)
