# Deployment

Two pieces deploy separately:

- **Backend (this repo)** → Hugging Face Spaces (Docker, free CPU).
- **Frontend (`scholar-rag-web`)** → Vercel.

The frontend calls the backend, and the backend allows the frontend's origin via
CORS — so deploy the **backend first**, then the frontend, then point CORS back.

---

## 1. Backend → Hugging Face Spaces

The app is a Docker Space listening on port **7860**. The prebuilt FAISS index
(`data/index/`, ~200 MB) is gitignored here, so it has to be added to the Space
repo directly via Git LFS.

### One-time setup

1. Create a Space: <https://huggingface.co/new-space> → **SDK: Docker**, name it
   e.g. `scholar-rag`. This creates a git repo at
   `https://huggingface.co/spaces/<user>/scholar-rag`.
2. Clone it and copy in the deploy files + the index:

   ```bash
   git clone https://huggingface.co/spaces/<user>/scholar-rag space && cd space

   # code + deploy recipe (from this repo)
   cp -r ../scholar-rag/src ../scholar-rag/pyproject.toml ../scholar-rag/uv.lock \
         ../scholar-rag/Dockerfile .

   # the prebuilt index, via LFS (it's large)
   git lfs install
   git lfs track "data/index/*"
   mkdir -p data && cp -r ../scholar-rag/data/index data/index
   ```

3. Add a `README.md` with the HF front-matter (HF needs this to know the port):

   ```
   ---
   title: Scholar RAG API
   emoji: 📚
   colorFrom: indigo
   colorTo: green
   sdk: docker
   app_port: 7860
   ---
   ```

4. In the Space UI → **Settings → Variables and secrets**, add:
   - `OPENROUTER_API_KEY` (secret) — your key.
   - `SCHOLAR_RAG_CORS_ORIGINS` (variable) — the Vercel URL (set after step 2 below),
     e.g. `https://scholar-rag-web.vercel.app`.

5. Commit & push:

   ```bash
   git add -A && git commit -m "Deploy scholar-rag API" && git push
   ```

HF builds the image and serves at `https://<user>-scholar-rag.hf.space`.
First boot downloads the embedding model and loads the index (~30–60 s).
Check `https://<user>-scholar-rag.hf.space/health`.

### Notes
- `OMP_NUM_THREADS=1` / `KMP_DUPLICATE_LIB_OK=TRUE` are baked into the Dockerfile
  (the faiss+torch OpenMP segfault guard).
- Feedback (`feedback.db`) is on the Space's **ephemeral** disk — it resets on
  rebuild/restart. Move to Turso/Supabase before relying on the data.

---

## 2. Frontend → Vercel

1. <https://vercel.com/new> → import the private `scholar-rag-web` repo.
   Vercel auto-detects Vite (build `bun run build` / `vite build`, output `dist`).
2. **Environment Variables** → add `VITE_API_BASE` = the HF Space URL
   (`https://<user>-scholar-rag.hf.space`), for Production.
3. Deploy. Vercel serves at `https://scholar-rag-web.vercel.app`.

## 3. Close the loop

Set the backend's `SCHOLAR_RAG_CORS_ORIGINS` (Space variable) to the Vercel URL
and restart the Space. Then the deployed frontend can call the deployed API.
