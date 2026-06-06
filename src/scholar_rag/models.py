from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

class RawDocument(BaseModel):
    """A whole document ingested and normalized by an adapter, before chunking.
    (source, source_doc_id) is the provenance key (for arXiv, the arXiv ID)."""
    source: str = Field(..., description="Adapter that produced this, e.g. 'arxiv'.")
    source_doc_id: str = Field(..., description="Stable ID within the source.")
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    full_text: str = Field(..., description="Extracted body text, ready to chunk.")
    categories: list[str] = Field(default_factory=list)
    published_at: datetime | None = None
    fetched_at: datetime = Field(default_factory=_utcnow)
    raw: dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    """A retrievable unit of text. (source, source_doc_id, chunk_index) is the
    idempotency key for upserts. Provenance on every chunk is what enables citations."""
    source: str
    source_doc_id: str
    chunk_index: int = Field(..., description="0-based position within the document")
    text: str
    section: str | None = None
    title: str | None = None
    categories: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None

    @property
    def chunk_id(self) -> str:
        return f"{self.source}:{self.source_doc_id}:{self.chunk_index}"

class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, paired with the score that surfaced it."""
    chunk: Chunk
    score: float
    retrieval_source: str = Field(default="unknown", description="e.g. 'dense', 'bm25', 'fused', 'reranked'.")


class Answer(BaseModel):
    """A generated answer grounded in retrieved context."""
    text: str
    citations: list[RetrievedChunk] = Field(default_factory=list)

