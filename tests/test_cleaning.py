from __future__ import annotations

from scholar_rag.chunking.cleaning import clean_text


def test_collapses_multiple_spaces():
    assert clean_text("a     b") == "a b"


def test_strips_trailing_whitespace_per_line():
    assert clean_text("line one   \nline two\t") == "line one\nline two"


def test_collapses_excessive_newlines():
    assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


def test_preserves_paragraph_breaks():
    assert clean_text("para one\n\npara two") == "para one\n\npara two"


def test_normalizes_carriage_returns():
    assert clean_text("a\r\nb\rc") == "a\nb\nc"


def test_preserves_short_and_numeric_content():
    text = "c1\n142\nn\nThe clique has 3 vertices."
    out = clean_text(text)
    assert "c1" in out and "142" in out and "n" in out