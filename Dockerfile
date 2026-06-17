# Backend image for Hugging Face Spaces (Docker SDK). The app listens on 7860,
# the port HF Spaces expects. Build context must include data/index/ (the
# prebuilt FAISS index) — it's gitignored, so see DEPLOY.md for how it travels.
FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# Install runtime deps from the lockfile (no dev tools, and not the project
# itself — scholar_rag is run via PYTHONPATH, it isn't a packaged dependency).
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project -o requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data/index ./data/index

ENV PYTHONPATH=/app/src \
    SCHOLAR_RAG_DATA_DIR=/app/data \
    HF_HOME=/tmp/hf \
    KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1
# OPENROUTER_API_KEY and SCHOLAR_RAG_CORS_ORIGINS are injected as Space secrets.

EXPOSE 7860
CMD ["uvicorn", "scholar_rag.serve.app:app", "--host", "0.0.0.0", "--port", "7860"]