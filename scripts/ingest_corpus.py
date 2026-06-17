"""Bulk, incremental, resumable ingest of data/pdfs/ into data/index/.

- Incremental: loads the existing index and skips docs already ingested
  (matched by chunk_id), so only new PDFs get embedded.
- Resumable: saves the index every SAVE_EVERY docs, so a crash (or Ctrl-C)
  loses at most that many docs — just re-run to continue.

Run: PYTHONPATH=src uv run python scripts/ingest_corpus.py
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from scholar_rag.chunking.chunker import RecursiveChunker
from scholar_rag.embedding.sentence_transformer import SentenceTransformerEmbedder
from scholar_rag.sources.local_pdf import LocalPdfSource
from scholar_rag.sources.validation import check_extraction
from scholar_rag.store.faiss_store import FaissVectorStore

logging.basicConfig(level=logging.ERROR, format="%(message)s")

PDF_DIR = Path("data/pdfs")
INDEX_DIR = Path("data/index")
SAVE_EVERY = 100  # docs between checkpoints

embedder = SentenceTransformerEmbedder()

if (INDEX_DIR / "index.faiss").exists():
    store = FaissVectorStore.load(INDEX_DIR, embedder)
    print(f"Loaded existing index: {len(store)} chunks", flush=True)
else:
    store = FaissVectorStore(embedder)
    print("Starting a fresh index", flush=True)

done = {c.source_doc_id for c in store.all_chunks()}

source = LocalPdfSource(PDF_DIR)
chunker = RecursiveChunker()
discovered = source.discover(max_results=10**9)
todo = [d for d in discovered if d.source_doc_id not in done]
print(f"{len(discovered)} PDFs found, {len(done)} already done, {len(todo)} to ingest", flush=True)

ingested = skipped = 0
t0 = time.time()

for i, d in enumerate(todo, start=1):
    try:
        doc = source.fetch(d.source_doc_id)
        check = check_extraction(doc)
        if not check.ok:
            skipped += 1
        else:
            store.add(chunker.chunk_document(doc))
            ingested += 1
    except Exception as e:  # one bad PDF shouldn't abort the whole run
        skipped += 1
        print(f"  ERROR {d.source_doc_id}: {e}", flush=True)

    if i % SAVE_EVERY == 0:
        store.save(INDEX_DIR)
        rate = i / (time.time() - t0)
        eta = (len(todo) - i) / rate if rate else 0
        print(
            f"  [{i}/{len(todo)}] ingested={ingested} skipped={skipped} "
            f"chunks={len(store)} ({rate:.1f} doc/s, ETA {eta/60:.1f} min)",
            flush=True,
        )

store.save(INDEX_DIR)
print(
    f"DONE: ingested {ingested}, skipped {skipped}, total chunks {len(store)} "
    f"in {(time.time() - t0)/60:.1f} min",
    flush=True,
)
