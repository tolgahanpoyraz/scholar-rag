# Scholar-RAG

**🔎 Live demo: [scholar-rag-web.vercel.app](https://scholar-rag-web.vercel.app)** &nbsp;·&nbsp; API: [HF Space](https://tpoyraz22-scholar-rag.hf.space/health)

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
  Fusion (`retrieval/`). BM25 is a **Rust extension** (`rust/`, via PyO3) using an
  inverted index. See [Performance](#performance).
- **Generation** — OpenRouter, prompted to cite passages as `[1]`, `[2, 3]` and
  to refuse when the context is insufficient (`generation/`).
- **API** — FastAPI, with token streaming and feedback capture (`serve/`).
- **Eval** — recall@k and MRR over a labeled query set (`eval.py`).

## Quickstart

```bash
uv sync

# build the Rust BM25 extension into the venv (required; rerun after Rust edits)
uv run maturin develop --release --manifest-path rust/Cargo.toml --uv

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

## Performance

BM25 was rewritten as a Rust extension (`rust/`, via PyO3). On the full corpus
(85,558 chunks, 10-core M-series, top-k=10):

| Implementation | Mean latency | Speedup |
|---|---:|---:|
| Pure Python, linear scan | ~57 ms | 1× |
| Rust, parallel linear scan | ~6 ms | ~9× |
| Rust, inverted index | 0.26 ms | ~220× |
| Rust, inverted index + stopwords | 0.097 ms | ~590× |

Most of the win is from the inverted index. It scores only the documents that
contain a query's terms instead of scanning all 85k, so the serial inverted index
beats the parallel full scan by ~27×. Dropping stopwords shrinks the rest. 
