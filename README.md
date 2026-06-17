# Scholar-RAG

A retrieval-augmented question-answering system over academic papers. Ask a
question in plain language and get a synthesized answer that is **grounded only
in the ingested corpus**, with inline citations back to the exact passage,
paper, and section. If the corpus can't answer, it says so rather than inventing
facts.

## How it works

```
PDFs / arXiv → ingest → chunk → embed → index
                                          │
query ─────────────────────────────────► retrieve (hybrid) → generate (cited) → answer
```

- **Sources** — local PDFs and arXiv (`sources/`).
- **Chunking** — section-aware recursive chunking (`chunking/`).
- **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2` (`embedding/`).
- **Vector store** — FAISS (`store/`); config has hooks for a Postgres + pgvector backend.
- **Retrieval** — **hybrid**: dense (FAISS) + BM25, fused with Reciprocal Rank
  Fusion (`retrieval/`).
- **Generation** — OpenRouter, prompted to cite passages as `[1]`, `[2, 3]` and
  to refuse when the context is insufficient (`generation/`).
- **API** — FastAPI, with token streaming and feedback capture (`serve/`).
- **Eval** — recall@k and MRR over a labeled query set (`eval.py`).

## Quickstart

```bash
uv sync
export OPENROUTER_API_KEY=...           # or put it in .env

# build the index from PDFs in data/pdfs/
PYTHONPATH=src uv run python scripts/ingest_corpus.py

# serve the API (OMP guards avoid a faiss+torch segfault on macOS)
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  PYTHONPATH=src uv run uvicorn scholar_rag.serve.app:app --env-file .env --port 8000
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness + library size |
| `POST /ask` | Cited answer (JSON) |
| `POST /ask/stream` | NDJSON token stream (`sources` → `token` → `done`) |
| `POST /feedback`, `DELETE /feedback/{id}` | Thumbs up/down, for finetuning data |

## Deployment

The API runs as a Docker image on Hugging Face Spaces; the web frontend is a
separate project deployed to Vercel. See [DEPLOY.md](DEPLOY.md).

## Roadmap

**Next up: a parallel BM25 retriever in Rust.** The current BM25 (`retrieval/bm25.py`)
is pure Python and scores a query by scanning *every* chunk, single-threaded —
about **59 ms mean / 17 queries-per-second** over an 85k-chunk corpus. Replacing
it with a Rust implementation (inverted index + parallelism) should cut query
latency by orders of magnitude. `scripts/bench_bm25.py` captures the Python
baseline to compare against.
