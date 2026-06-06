from __future__ import annotations

from scholar_rag.chunking.chunker import RecursiveChunker
from scholar_rag.ingest import Ingestor
from scholar_rag.models import RawDocument
from scholar_rag.sources.base import DiscoveredDoc, DocumentSource
from scholar_rag.store.faiss_store import FaissVectorStore
from tests.fakes import FakeEmbedder


class FakeSource(DocumentSource):
    name = "fake"

    def __init__(self, docs: dict[str, RawDocument], titles: dict[str, str]):
        self._docs = docs
        self._titles = titles

    def discover(self, query="", max_results=50):
        return [
            DiscoveredDoc(source=self.name, source_doc_id=did, title=self._titles[did],
                          authors=[], abstract=None, categories=[])
            for did in self._docs
        ]

    def fetch(self, doc_id):
        return self._docs[doc_id]


def _chunker():
    from tests.test_chunker import WordCounter
    return RecursiveChunker(target_tokens=20, overlap_tokens=0,
                            counter=WordCounter(), section_aware=False)


def _source():
    body = " ".join(f"word{i}" for i in range(60))   # enough to chunk
    docs = {
        "good1": RawDocument(source="fake", source_doc_id="good1", title="", full_text=body),
        "good2": RawDocument(source="fake", source_doc_id="good2", title="", full_text=body),
        "scanned": RawDocument(source="fake", source_doc_id="scanned", title="", full_text=""),
    }
    titles = {"good1": "First Paper", "good2": "Second Paper", "scanned": "Scanned Paper"}
    return FakeSource(docs, titles)


def test_ingests_valid_docs_and_skips_empty():
    store = FaissVectorStore(FakeEmbedder())
    report = Ingestor(_source(), _chunker(), store).ingest_all()
    assert set(report.ingested) == {"good1", "good2"}
    assert len(report.skipped) == 1
    assert report.skipped[0][0] == "scanned"
    assert report.total_chunks > 0
    assert len(store) == report.total_chunks


def test_title_backfilled_from_discovery():
    store = FaissVectorStore(FakeEmbedder())
    Ingestor(_source(), _chunker(), store).ingest_all()
    titles = {c.title for c in store._chunks}
    assert "First Paper" in titles and "Second Paper" in titles


def test_idempotent_reingest():
    store = FaissVectorStore(FakeEmbedder())
    ing = Ingestor(_source(), _chunker(), store)
    ing.ingest_all()
    n = len(store)
    ing.ingest_all()
    assert len(store) == n