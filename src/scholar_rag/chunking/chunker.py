from __future__ import annotations

from scholar_rag.chunking.tokenizer import TiktokenCounter, TokenCounter
from scholar_rag.models import Chunk, RawDocument

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

class RecursiveChunker:
    def __init__(
            self,
            target_tokens: int = 250,
            overlap_tokens: int = 40,
            counter: TokenCounter | None = None,
            separators: list[str] | None = None,
    ) -> None:
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap must be smaller than target")
        self._target = target_tokens
        self._overlap = overlap_tokens
        self._counter = counter or TiktokenCounter()
        self._separators = separators or _DEFAULT_SEPARATORS

    def chunk_document(self, doc: RawDocument) -> list[Chunk]:
        pieces = self._split(doc.full_text, self._separators)
        merged = self._merge_with_overlap(pieces)
        return [
            Chunk(
                source=doc.source,
                source_doc_id=doc.source_doc_id,
                chunk_index=i,
                text=text,
                title=doc.title or None,
                categories=doc.categories,
            )
            for i, text in enumerate(merged)
        ]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if self._counter.count(text) <= self._target:
            return [text] if text.strip() else []

        sep, *rest = separators
        if sep == "":
            return self._slice_oversized(text)

        parts = text.split(sep)
        out: list[str] = []
        for part in parts:
            part = part + sep
            if self._counter.count(part) <= self._target:
                if part.strip():
                    out.append(part)
            else:
                out.extend(self._split(part, rest))
        return out

    def _slice_oversized(self, blob: str) -> list[str]:
        ratio = max(1, len(blob)) // max(1, self._counter.count(blob))
        span = max(1, (self._target * ratio) // 2)
        return [blob[i:i + span] for i in range(0, len(blob), span)]

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        if not pieces:
            return []
        packed: list[str] = []
        cur = ""
        for p in pieces:
            candidate = (cur + p) if cur else p
            if self._counter.count(candidate) <= self._target:
                cur = candidate
            else:
                if cur:
                    packed.append(cur)
                cur = p
        if cur:
            packed.append(cur)

        if self._overlap == 0 or len(packed) <= 1:
            return packed
        result = [packed[0]]
        for i in range(1, len(packed)):
            tail = self._tail_tokens(packed[i - 1], self._overlap)
            result.append((tail + " " + packed[i]).strip())
        return result

    def _tail_tokens(self, text: str, n_tokens: int) -> str:
        words = text.split(" ")
        tail: list[str] = []
        for w in reversed(words):
            tail.insert(0, w)
            if self._counter.count(" ".join(tail)) >= n_tokens:
                break
        return " ".join(tail)