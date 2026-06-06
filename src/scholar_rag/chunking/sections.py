from __future__ import annotations

import re
from dataclasses import dataclass

_SECTION_WORDS = {
    "abstract", "introduction", "related work", "background", "preliminaries",
    "notation", "methods", "methodology", "results", "discussion",
    "conclusion", "conclusions", "references", "acknowledgement",
    "acknowledgements", "acknowledgment", "acknowledgments", "appendix",
    "keywords", "bibliography",
}

# Optional leading number ('1.', '3.1', '2') or roman numeral ('II.') + whitespace.
_PREFIX_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+|[IVXLC]+\.?\s+)")
# Optional trailing '.' or ':' and any whitespace, anchored at end of line.
_TRAILING_RE = re.compile(r"[.:]?\s*$")

@dataclass(frozen=True)
class SectionMarker:
    """A detected header: its canonical name and char offest in the source text."""
    name: str
    offset: int

def _normalize(line: str) -> str | None:
    s = line.strip()
    if not s or len(s) > 50:
        return None
    core = _PREFIX_RE.sub("", s)
    core = _TRAILING_RE.sub("", core).strip().lower()
    return core if core in _SECTION_WORDS else None

def find_sections(text: str) -> list[SectionMarker]:
    """Scan `text` line by line, returning section headers with their char offsets in document order"""
    markers: list[SectionMarker] = []
    offset = 0
    for line in text.split("\n"):
        name = _normalize(line)
        if name is not None:
            markers.append(SectionMarker(name=name, offset=offset))
        offset += len(line) + 1     # +1 for the '\n' that split() consumed
    return markers

def split_into_sections(text: str) -> list[tuple[str | None, str]]:
    markers = find_sections(text)
    if not markers:
        return [(None, text)]

    segments: list[tuple[str | None, str]] = []
    if markers[0].offset > 0:
        segments.append((None, text[: markers[0].offset]))
    for i, m in enumerate(markers):
        end = markers[i + 1].offset if i + 1 < len(markers) else len(text)
        segments.append((m.name, text[m.offset : end]))
    return segments
