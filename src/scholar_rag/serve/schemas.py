"""The HTTP contract: what the API accepts and returns, as JSON.

These models are deliberately *separate* from the domain models in
scholar_rag/models.py. The domain models describe what the system knows; these
describe what the frontend needs to render. Keeping them apart means we can
reshape the API for the UI without disturbing retrieval/generation internals.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from scholar_rag.models import RetrievedChunk


class AskRequest(BaseModel):
    """A question from the user."""

    question: str = Field(..., min_length=1, max_length=2000)


class SourceOut(BaseModel):
    """One cited passage, flattened for the UI.

    `index` is the 1-based citation number: it matches the [1], [2] markers in
    the answer text, and is how the frontend links a marker to its card.
    """

    index: int
    title: str | None
    authors: list[str]
    source: str
    doc_id: str
    section: str | None
    retrieval: str  # "dense" | "bm25" | "fused" — the UI maps these to labels
    score: float
    text: str

    @classmethod
    def from_retrieved(cls, index: int, rc: RetrievedChunk) -> "SourceOut":
        c = rc.chunk
        return cls(
            index=index,
            title=c.title,
            authors=[],  # not stored on Chunk yet — wired through later
            source=c.source,
            doc_id=c.source_doc_id,
            section=c.section,
            retrieval=rc.retrieval_source,
            score=rc.score,
            text=c.text,
        )


class AskResponse(BaseModel):
    """The full answer payload the UI renders.

    `answer_id` identifies this specific generated answer in storage; the UI
    sends it back with any thumbs up/down so the vote attaches to the
    answer that was rated.
    """

    question: str
    answer_id: str
    answer: str
    sources: list[SourceOut]


class FeedbackRequest(BaseModel):
    """A thumbs up/down on a previously returned answer."""

    answer_id: str
    vote: Literal["up", "down"]
    reasons: list[str] = Field(default_factory=list)
    comment: str | None = Field(default=None, max_length=2000)
