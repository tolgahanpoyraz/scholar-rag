from __future__ import annotations

from scholar_rag.generation.prompt import SYSTEM_PROMPT, build_user_prompt
from scholar_rag.models import Chunk, RetrievedChunk


def _rc(doc_id: str, section, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(source="arxiv", source_doc_id=doc_id, chunk_index=0,
                    text=text, section=section),
        score=0.5,
    )


def test_prompt_numbers_passages_and_includes_query():
    chunks = [_rc("d1", "introduction", "first passage"),
              _rc("d2", None, "second passage")]
    p = build_user_prompt("what is X?", chunks)
    assert "[1]" in p and "[2]" in p
    assert "first passage" in p and "second passage" in p
    assert "what is X?" in p


def test_prompt_includes_provenance():
    p = build_user_prompt("q", [_rc("2401.01234", "results", "txt")])
    assert "2401.01234" in p
    assert "results" in p


def test_prompt_handles_empty_context():
    p = build_user_prompt("q", [])
    assert "no relevant passages" in p
    assert "q" in p


def test_system_prompt_enforces_grounding_and_citation():
    s = SYSTEM_PROMPT.lower()
    assert "only" in s
    assert "cite" in s
    assert "do not invent" in s