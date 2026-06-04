from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from scholar_rag.sources.local_pdf import LocalPdfSource

def _make_pdf(path: Path, text: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()

@pytest.fixture
def pdf_folder(tmp_path: Path) -> Path:
    _make_pdf(tmp_path / "erdos_1964.pdf", "On the reconstruction of graphs.")
    _make_pdf(tmp_path / "split_graphs.pdf", "A split graph partitions into a clique.")
    return tmp_path

def test_discover_lists_all_pdfs(pdf_folder: Path):
    src = LocalPdfSource(pdf_folder)
    docs = src.discover()
    ids = {d.source_doc_id for d in docs}
    assert ids == {"erdos_1964", "split_graphs"}
    assert all(d.source == "local_pdf" for d in docs)

def test_discover_filters_by_query(pdf_folder: Path):
    src = LocalPdfSource(pdf_folder)
    docs = src.discover(query="split")
    assert [d.source_doc_id for d in docs] == ["split_graphs"]

def test_fetch_extracts_text(pdf_folder: Path):
    src = LocalPdfSource(pdf_folder)
    doc = src.fetch("erdos_1964")
    assert doc.source == "local_pdf"
    assert doc.source_doc_id == "erdos_1964"
    assert "reconstruction" in doc.full_text.lower()

def test_fetch_missing_raises(pdf_folder: Path):
    src = LocalPdfSource(pdf_folder)
    with pytest.raises(FileNotFoundError):
        src.fetch("does_not_exist")

def test_bad_folder_raises():
    with pytest.raises(NotADirectoryError):
        LocalPdfSource("/no/such/folder/here")