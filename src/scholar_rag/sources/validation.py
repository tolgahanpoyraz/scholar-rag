from __future__ import annotations
from dataclasses import dataclass
from scholar_rag.models import RawDocument

_MIN_CHARS = 200

@dataclass(frozen=True)
class ExtractionCheck:
    """Result of validating a document's extracted text. `ok=False` means
    the caller should skip+log, not crash."""
    ok: bool
    char_count: int
    reason: str | None = None

def check_extraction(doc: RawDocument, min_chars: int = _MIN_CHARS) -> ExtractionCheck:
    n = len(doc.full_text.strip())
    if n < min_chars:
        return ExtractionCheck(
            ok=False,
            char_count=n,
            reason=(
                f"only {n} chars extracted (< {min_chars}); "
                f"likely a scanned/image PDF needing OCR"
            ),
        )
    return ExtractionCheck(ok=True, char_count=n)