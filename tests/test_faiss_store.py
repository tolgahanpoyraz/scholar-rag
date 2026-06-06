from __future__ import annotations

from pathlib import Path

from scholar_rag.models import Chunk
from scholar_rag.store.faiss_store import FaissVectorStore
from tests.fakes import FakeEmbedder


def _chunk(idx: int, text: str) -> Chunk:
    return Chunk(source="t", source_doc_id="d1", chunk_index=idx, text=text)


def _store() -> FaissVectorStore:
    s = FaissVectorStore(FakeEmbedder(dimension=16))
    s.add([_chunk(0, "alpha text about clauses"),
           _chunk(1, "beta text about graphs"),
           _chunk(2, "gamma text about numbers")])
    return s


def test_add_increases_length():
    s = _store()
    assert len(s) == 3


def test_search_empty_store_returns_empty():
    s = FaissVectorStore(FakeEmbedder())
    assert s.search("anything") == []


def test_search_returns_self_first():
    s = _store()
    results = s.search("beta text about graphs", k=3)
    assert results[0].chunk.text == "beta text about graphs"
    assert results[0].score > 0.99
    assert results[0].retrieval_source == "dense"


def test_search_results_sorted_descending():
    s = _store()
    results = s.search("alpha text about clauses", k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_k_larger_than_store_is_safe():
    s = _store()
    results = s.search("alpha text about clauses", k=100)  # only 3 chunks
    assert len(results) == 3


def test_idempotent_add():
    s = _store()
    s.add([_chunk(0, "alpha text about clauses")])  # same chunk_id as before
    assert len(s) == 3


def test_save_load_roundtrip(tmp_path: Path):
    s = _store()
    s.save(tmp_path)
    loaded = FaissVectorStore.load(tmp_path, FakeEmbedder(dimension=16))
    assert len(loaded) == 3
    results = loaded.search("beta text about graphs", k=1)
    assert results[0].chunk.text == "beta text about graphs"
    assert results[0].score > 0.99