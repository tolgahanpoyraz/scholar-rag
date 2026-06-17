"""The FastAPI application: HTTP in, JSON out.

This is the only layer that knows about the web. It builds one RagEngine at
startup, then translates HTTP requests into engine calls and engine results
into the JSON contract from schemas.py.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from scholar_rag.config import load_config
from scholar_rag.serve.engine import RagEngine
from scholar_rag.serve.feedback_store import FeedbackStore
from scholar_rag.serve.schemas import (
    AskRequest,
    AskResponse,
    FeedbackRequest,
    SourceOut,
)

logger = logging.getLogger("scholar_rag.serve")

# Browsers block cross-origin calls unless the server opts in. In dev the React
# app runs on localhost:5173 (Vite); in prod it'll be your GitHub Pages origin.
# Set SCHOLAR_RAG_CORS_ORIGINS as a comma-separated list to override.
_DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the engine and open the feedback store, before accepting traffic."""
    config = load_config()
    logger.info("Building RAG engine (loading model + index)...")
    app.state.engine = RagEngine.from_config(config)
    app.state.feedback = FeedbackStore(config.data_dir / "feedback.db")
    logger.info("Engine ready: %d papers in library.", app.state.engine.library_size)
    yield
    # nothing to tear down


app = FastAPI(title="scholar-rag", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("SCHOLAR_RAG_CORS_ORIGINS", _DEFAULT_ORIGINS).split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Liveness + the library size the UI header displays."""
    engine: RagEngine = app.state.engine
    return {"status": "ok", "library_size": engine.library_size}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Answer one question with cited passages, and persist it for rating."""
    engine: RagEngine = app.state.engine
    store: FeedbackStore = app.state.feedback
    try:
        answer = engine.ask(req.question)
    except Exception:
        # Retrieval or the LLM call failed. Surface a clean 502 so the frontend
        # can show its "Something went wrong reaching the index" error state,
        # rather than leaking a stack trace. We log the real cause server-side.
        logger.exception("ask() failed for question=%r", req.question)
        raise HTTPException(status_code=502, detail="retrieval_or_generation_failed")

    sources = [
        SourceOut.from_retrieved(i, rc)
        for i, rc in enumerate(answer.citations, start=1)
    ]
    answer_id = store.record_answer(
        req.question, answer.text, [s.model_dump() for s in sources]
    )
    return AskResponse(
        question=req.question,
        answer_id=answer_id,
        answer=answer.text,
        sources=sources,
    )


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Same as /ask, but streamed as NDJSON events: one `sources` event, many
    `token` events, then a final `done` event with the persisted answer_id."""
    engine: RagEngine = app.state.engine
    store: FeedbackStore = app.state.feedback

    def event(obj: dict) -> str:
        return json.dumps(obj) + "\n"

    def emit():
        try:
            chunks = engine.retrieve(req.question)
        except Exception:
            logger.exception("retrieve failed for %r", req.question)
            yield event({"type": "error", "detail": "retrieval_failed"})
            return

        sources = [SourceOut.from_retrieved(i, rc) for i, rc in enumerate(chunks, start=1)]
        source_dicts = [s.model_dump() for s in sources]
        yield event({"type": "sources", "sources": source_dicts})

        parts: list[str] = []
        try:
            for delta in engine.stream(req.question, chunks):
                parts.append(delta)
                yield event({"type": "token", "text": delta})
        except Exception:
            logger.exception("stream generation failed for %r", req.question)
            yield event({"type": "error", "detail": "generation_failed"})
            return

        # Persist the finished answer (same record as /ask) and hand back its id.
        answer_id = store.record_answer(req.question, "".join(parts), source_dicts)
        yield event({"type": "done", "answer_id": answer_id})

    return StreamingResponse(emit(), media_type="application/x-ndjson")


@app.post("/feedback", status_code=204)
def submit_feedback(req: FeedbackRequest) -> Response:
    """Attach (or replace) a thumbs up/down on a previously returned answer."""
    store: FeedbackStore = app.state.feedback
    ok = store.record_feedback(req.answer_id, req.vote, req.reasons, req.comment)
    if not ok:
        raise HTTPException(status_code=404, detail="unknown_answer_id")
    return Response(status_code=204)


@app.delete("/feedback/{answer_id}", status_code=204)
def undo_feedback(answer_id: str) -> Response:
    """Remove a vote (the UI's 'undo'). Idempotent."""
    store: FeedbackStore = app.state.feedback
    store.delete_feedback(answer_id)
    return Response(status_code=204)
