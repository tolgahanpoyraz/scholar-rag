from __future__ import annotations

import logging
from dataclasses import dataclass, field

from scholar_rag.chunking.chunker import RecursiveChunker
from scholar_rag.sources.base import DocumentSource
from scholar_rag.sources.validation import check_extraction
from scholar_rag.store.base import VectorStore

log = logging.getLogger(__name__)

@dataclass
class IngestReport:
    ingested: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)    # (doc_id, reason)
    total_chunks: int = 0

    def summary(self) -> str:
        return (f"ingested {len(self.ingested)} docs "
                f"({self.total_chunks} chunks), skipped {len(self.skipped)}")

class Ingestor:
    def __init__(
            self,
            source: DocumentSource,
            chunker: RecursiveChunker,
            store: VectorStore,
    ) -> None:
        self._source = source
        self._chunker = chunker
        self._store = store

    def ingest_all(self, query: str = "", max_results: int = 100) -> IngestReport:
        report = IngestReport()
        for d in self._source.discover(query, max_results):
            doc = self._source.fetch(d.source_doc_id)

            doc = doc.model_copy(
                update={
                    "title": doc.title or d.title,
                    "authors": doc.authors or d.authors,
                    "abstract": doc.abstract or d.abstract,
                    "categories": doc.categories or d.categories,
                }
            )

            check = check_extraction(doc)
            if not check.ok:
                log.warning("skipping %s: %s", d.source_doc_id, check.reason)
                report.skipped.append((d.source_doc_id, check.reason or "Invalid"))
                continue

            chunks = self._chunker.chunk_document(doc)
            self._store.add(chunks)
            report.ingested.append(d.source_doc_id)
            report.total_chunks += len(chunks)

        return report

