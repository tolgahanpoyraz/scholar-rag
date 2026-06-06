from __future__ import annotations

from scholar_rag.chunking.sections import (
    find_sections,
    split_into_sections,
)
from scholar_rag.chunking.sections import _normalize


def test_matches_plain_headers():
    assert _normalize("Abstract") == "abstract"
    assert _normalize("Introduction") == "introduction"
    assert _normalize("References") == "references"


def test_matches_allcaps():
    assert _normalize("ABSTRACT") == "abstract"
    assert _normalize("KEYWORDS") == "keywords"


def test_matches_numbered_and_roman_prefix():
    assert _normalize("1. Introduction") == "introduction"
    assert _normalize("1. Introduction.") == "introduction"
    assert _normalize("II. Methods") == "methods"


def test_matches_trailing_punctuation():
    assert _normalize("Abstract.") == "abstract"
    assert _normalize("Discussion:") == "discussion"


def test_matches_multiword():
    assert _normalize("Related Work") == "related work"


def test_rejects_sentence_starting_with_section_word():
    assert _normalize("Results on the edge-reconstruction number of a") is None


def test_rejects_numbered_theorem_and_case():
    assert _normalize("1. If not all components of G are isomorphic, then ern(") is None
    assert _normalize("1. Family 1 (|C| = 2, balanced double star)") is None


def test_rejects_noise():
    assert _normalize("AUSTRALASIAN JOURNAL OF COMBINATORICS") is None
    assert _normalize("772") is None
    assert _normalize("") is None
    assert _normalize("   ") is None

def test_find_sections_offsets_are_correct():
    text = "Title\n\nIntroduction\nbody text\n\nConclusion\nmore"
    markers = find_sections(text)
    assert [m.name for m in markers] == ["introduction", "conclusion"]
    for m in markers:
        line_at_offset = text[m.offset:].split("\n", 1)[0]
        assert _normalize(line_at_offset) == m.name


def test_find_sections_empty_when_none():
    assert find_sections("just some prose\nwith no headers at all") == []


def test_split_labels_segments():
    text = "Title here\n\nIntroduction\nintro body\n\nConclusion\nfinal words"
    segs = split_into_sections(text)
    names = [name for name, _ in segs]
    assert names == [None, "introduction", "conclusion"]
    intro_text = dict((n, t) for n, t in segs if n)["introduction"]
    assert "intro body" in intro_text
    assert "final words" not in intro_text


def test_split_no_headers_is_single_none_segment():
    text = "prose with no recognizable section headers whatsoever"
    assert split_into_sections(text) == [(None, text)]


def test_split_no_preamble_when_header_is_first():
    text = "Abstract\nthe abstract body"
    segs = split_into_sections(text)
    assert segs[0][0] == "abstract"