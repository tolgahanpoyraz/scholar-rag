from __future__ import annotations

from scholar_rag.chunking.chunker import RecursiveChunker
from scholar_rag.chunking.tokenizer import TokenCounter, TiktokenCounter
from scholar_rag.models import RawDocument
from scholar_rag.chunking.cleaning import clean_text

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

def test_section_aware_tags_chunks():
    text = ("Title Line\n\nAbstract\nthe abstract body here\n\n"
            "Introduction\nthe introduction body here\n\n"
            "Conclusion\nthe conclusion body")
    doc = RawDocument(source="arxiv", source_doc_id="d1", title="T", full_text=text)
    chunker = RecursiveChunker(target_tokens=8, overlap_tokens=0, counter=WordCounter())
    chunks = chunker.chunk_document(doc)
    sections = {c.section for c in chunks}
    assert "abstract" in sections
    assert "introduction" in sections
    assert "conclusion" in sections
    assert None in sections


def test_baseline_mode_has_no_sections():
    text = "Abstract\nbody\n\nIntroduction\nmore body content here for splitting"
    doc = RawDocument(source="arxiv", source_doc_id="d1", title="T", full_text=text)
    chunker = RecursiveChunker(target_tokens=8, overlap_tokens=0,
                               counter=WordCounter(), section_aware=False)
    chunks = chunker.chunk_document(doc)
    assert all(c.section is None for c in chunks)

def test_no_headers_falls_back_to_baseline():
    text = "just prose with no recognizable headers anywhere in this body of text"
    doc = RawDocument(source="t", source_doc_id="d1", title="T", full_text=text)
    chunks = RecursiveChunker(counter=WordCounter()).chunk_document(doc)
    assert all(c.section is None for c in chunks)


def test_chunk_indices_global_across_sections():
    text = "Abstract\na b c d e f\n\nIntroduction\ng h i j k l"
    doc = RawDocument(source="t", source_doc_id="d1", title="T", full_text=text)
    chunker = RecursiveChunker(target_tokens=3, overlap_tokens=0, counter=WordCounter())
    chunks = chunker.chunk_document(doc)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

def test_cleaning_applied_in_chunking():
    doc = RawDocument(source="t", source_doc_id="d1", title="T",
                      full_text="word1     word2\t\tword3")
    chunks = RecursiveChunker(counter=WordCounter()).chunk_document(doc)
    assert "     " not in chunks[0].text

def test_section_label_capped_for_oversized_segment():
    abstract = "Abstract\n" + " ".join(f"a{i}" for i in range(5))
    intro = "Introduction\n" + " ".join(f"w{i}" for i in range(200))
    doc = RawDocument(source="t", source_doc_id="d1", title="T",
                      full_text=f"{abstract}\n\n{intro}")
    chunker = RecursiveChunker(target_tokens=10, overlap_tokens=0,
                               counter=WordCounter(), max_section_tokens=30)
    chunks = chunker.chunk_document(doc)
    intro_chunks = [c for c in chunks if "w0" in c.text or "w5" in c.text
                    or any(f"w{i}" in c.text for i in range(200))]
    labels = [c.section for c in intro_chunks]
    assert "introduction" in labels
    assert None in labels

def test_short_section_keeps_label_fully():
    text = "Abstract\n" + " ".join(f"a{i}" for i in range(10))  # well under cap
    doc = RawDocument(source="t", source_doc_id="d1", title="T", full_text=text)
    chunker = RecursiveChunker(target_tokens=5, overlap_tokens=0,
                               counter=WordCounter(), max_section_tokens=1000)
    chunks = chunker.chunk_document(doc)
    abstract_chunks = [c for c in chunks if c.section == "abstract"]
    assert len(abstract_chunks) == len([c for c in chunks if "a" in c.text])