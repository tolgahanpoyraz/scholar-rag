from __future__ import annotations

import os

import pytest

from scholar_rag.models import Chunk, RetrievedChunk
from tests.fakes import FakeGenerator


def _rc(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(source="t", source_doc_id="d1", chunk_index=0, text=text),
        score=0.5,
    )


def test_fake_generator_returns_answer_with_citations():
    chunks = [_rc("passage one"), _rc("passage two")]
    answer = FakeGenerator().generate("what is X?", chunks)
    assert "what is X?" in answer.text
    assert answer.citations == chunks
    assert "[1]" in answer.text and "[2]" in answer.text


def test_fake_generator_handles_no_chunks():
    answer = FakeGenerator().generate("q", [])
    assert answer.citations == []
    assert "(none)" in answer.text


def test_openrouter_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from scholar_rag.generation.openrouter import OpenRouterGenerator
    with pytest.raises(ValueError, match="API key"):
        OpenRouterGenerator()


@pytest.mark.slow
def test_openrouter_real_call():
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")
    from scholar_rag.generation.openrouter import OpenRouterGenerator
    answer = OpenRouterGenerator().generate(
        "What is a split graph?",
        [_rc("A split graph is one whose vertices partition into a clique "
             "and an independent set.")],
    )
    assert len(answer.text) > 0
    assert answer.citations