from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from scholar_rag.models import RawDocument
from scholar_rag.sources.base import DiscoveredDoc, DocumentSource
from scholar_rag.sources.pdf_text import extract_text_from_path

class LocalPdfSource(DocumentSource):
    name = "local_pdf"

    def __init__(self, folder: Path | str) -> None:
        self._folder = Path(folder)
        if not self._folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {self._folder}")

    def _path_for(self, doc_id: str) -> Path:
        return self._folder / f"{doc_id}.pdf"

    def discover(self, query: str = "", max_results: int = 50) -> list[DiscoveredDoc]:
        pdfs = sorted(self._folder.glob("*.pdf"))
        if query:
            q = query.lower()
            pdfs = [p for p in pdfs if q in p.stem.lower()]
        out: list[DiscoveredDoc] = []
        for p in pdfs[:max_results]:
            out.append(
                DiscoveredDoc(
                    source=self.name,
                    source_doc_id=p.stem,
                    title=p.stem,
                    authors=[],
                    abstract=None,
                    categories=[],
                )
            )
        return out

    def fetch(self, doc_id: str) -> RawDocument:
        path = self._path_for(doc_id)
        if not path.is_file():
            raise FileNotFoundError(f"No such PDF: {path}")
        return RawDocument(
            source=self.name,
            source_doc_id=doc_id,
            title=doc_id,
            full_text=extract_text_from_path(path),
            fetched_at=datetime.now(timezone.utc),
            raw={"path": str(path)}
        )