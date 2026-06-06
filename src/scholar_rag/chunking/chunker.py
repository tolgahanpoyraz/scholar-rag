from __future__ import annotations

from scholar_rag.chunking.tokenizer import TiktokenCounter, TokenCounter
from scholar_rag.models import Chunk, RawDocument
from scholar_rag.chunking.cleaning import clean_text
from scholar_rag.chunking.sections import split_into_sections

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

class RecursiveChunker:
    def __init__(
            self,
            target_tokens: int = 250,
            overlap_tokens: int = 40,
            counter: TokenCounter | None = None,
            separators: list[str] | None = None,
            section_aware: bool = True,
            clean: bool = True,
            max_section_tokens: int = 2000,
    ) -> None:
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap must be smaller than target")
        self._target = target_tokens
        self._overlap = overlap_tokens
        self._counter = counter or TiktokenCounter()
        self._separators = separators or _DEFAULT_SEPARATORS
        self._section_aware = section_aware
        self._clean = clean
        self._max_section_tokens = max_section_tokens

    def chunk_document(self, doc: RawDocument) -> list[Chunk]:
        text = clean_text(doc.full_text) if self._clean else doc.full_text

        segments = split_into_sections(text) if self._section_aware else [(None, text)]

        chunks: list[Chunk] = []
        idx = 0
        for section_name, seg_text in segments:
            pieces = self._split(seg_text, self._separators)
            merged = self._merge_with_overlap(pieces)
            tokens_in_section = 0
            for piece_text in merged:
                label = section_name
                if section_name is not None and tokens_in_section >= self._max_section_tokens:
                    label = None
                chunks.append(
                    Chunk(
                        source=doc.source,
                        source_doc_id=doc.source_doc_id,
                        chunk_index=idx,
                        text=piece_text,
                        section=label,
                        title=doc.title or None,
                        categories=doc.categories,
                    )
                )
                tokens_in_section += self._counter.count(piece_text)
                idx += 1
        return chunks

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