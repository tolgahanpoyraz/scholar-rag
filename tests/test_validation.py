from __future__ import annotations

from scholar_rag.models import RawDocument
from scholar_rag.sources.validation import check_extraction

def _doc(text: str) -> RawDocument:
    return RawDocument(source="test", source_doc_id="d", title="T", full_text=text)

def test_normal_doc_passes():
    check = check_extraction(_doc("x" * 5000))
    assert check.ok
    assert check.char_count == 5000
    assert check.reason is None

def test_empty_doc_flagged():
    check = check_extraction(_doc(""))
    assert not check.ok
    assert check.char_count == 0
    assert "scanned" in check.reason

def test_whitespace_only_flagged():
    check = check_extraction(_doc("   \n\n  \t  "))
    assert not check.ok
    assert check.char_count == 0

def test_threshold_is_configurable():
    doc = _doc("x" * 150)
    assert check_extraction(doc, min_chars=100).ok
    assert not check_extraction(doc, min_chars=200).ok
