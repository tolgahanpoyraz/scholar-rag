from __future__ import annotations

from scholar_rag.chunking.chunker import RecursiveChunker
from scholar_rag.chunking.tokenizer import TokenCounter, TiktokenCounter
from scholar_rag.models import RawDocument

class WordCounter(TokenCounter):
    def count(self, text: str) -> int:
        return len(text.split())

def _doc(text: str) -> RawDocument:
    return RawDocument(source="test", source_doc_id="d1", title="T", full_text=text)

def test_short_text_is_single_chunk():
    chunker = RecursiveChunker(target_tokens=50, overlap_tokens=0, counter=WordCounter())
    chunks = chunker.chunk_document(_doc("a short sentence with few words"))
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].source == "test"

def test_long_text_splits_into_multiple_chunks():
    text = " ".join(f"words{i}" for i in range(100))
    chunker = RecursiveChunker(target_tokens=20, overlap_tokens=0, counter=WordCounter())
    chunks = chunker.chunk_document(_doc(text))
    assert len(chunks) >= 4
    assert all(WordCounter().count(c.text) <= 20 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

def test_overlap_shares_boundary_text():
    text = " ".join(f"w{i}" for i in range(60))
    chunker = RecursiveChunker(target_tokens=20, overlap_tokens=5, counter=WordCounter())
    chunks = chunker.chunk_document(_doc(text))
    first_tail = chunks[0].text.split()[-3:]
    assert all(w in chunks[1].text for w in first_tail)

def test_prefers_paragraph_boundaries():
    text = "alpha beta gamma\n\ndelta epsilon zeta"
    chunker = RecursiveChunker(target_tokens=4, overlap_tokens=0, counter=WordCounter())
    chunks = chunker.chunk_document(_doc(text))
    assert any("alpha" in c.text and "gamma" in c.text and "delta" not in c.text for c in chunks)

def test_provenance_and_metadata_carried_through():
    doc = RawDocument(source="arxiv", source_doc_id="2401.01234", title="My Paper",
                      full_text="content here", categories=["math.CO"])
    chunks = RecursiveChunker(counter=WordCounter()).chunk_document(doc)
    c = chunks[0]
    assert c.source == "arxiv"
    assert c.source_doc_id == "2401.01234"
    assert c.title == "My Paper"
    assert c.categories == ["math.CO"]
    assert c.chunk_id == "arxiv:2401.01234:0"

def test_overlap_must_be_smaller_than_target():
    import pytest
    with pytest.raises(ValueError):
        RecursiveChunker(target_tokens=10, overlap_tokens=10, counter=WordCounter())

def test_tiktoken_counter_is_reasonable():
    counter = TiktokenCounter()
    assert counter.count("") == 0
    assert counter.count("hello world") < counter.count("hello world foo bar baz qux")