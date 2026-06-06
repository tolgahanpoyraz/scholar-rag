from __future__ import annotations

from scholar_rag.embedding.base import Embedder

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = _DEFAULT_MODEL, normalize: bool = True) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._normalize = normalize


    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
