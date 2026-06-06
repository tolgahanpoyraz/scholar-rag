from __future__ import annotations

from scholar_rag.models import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a research assistant answering questions about academic papers. "
    "Answer using ONLY the provided context passages. Cite the passages you rely "
    "on with bracketed numbers like [1] or [2, 3]. If the passages do not contain "
    "enough information to answer the question, say so plainly — do not invent "
    "facts or draw on outside knowledge."
)

def build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        blocks = "(no relevant passages found)"
    else:
        blocks = "\n\n".join(
            f"[{i}] (source: {rc.chunk.source_doc_id}"
            + (f", {rc.chunk.section}" if rc.chunk.section else "")
            + f")\n{rc.chunk.text}"
            for i, rc in enumerate(chunks, 1)
        )
    return f"Context passages:\n\n{blocks}\n\nQuestion: {query}"