"""Central config, read from the environment.
Unset DATABASE_URL -> local SQLite+FAISS. Set it -> Postgres+pgvector"""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    data_dir: Path
    database_url: str | None
    embedding_model: str
    top_k: int

    @property
    def is_cloud(self) -> bool:
        return self.database_url is not None

def load_config() -> Config:
    data_dir = Path(os.environ.get("SCHOLAR_RAG_DATA_DIR", "data")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        data_dir=data_dir,
        database_url=os.environ.get("DATABASE_URL"),
        embedding_model=os.environ.get("SCHOLAR_RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        top_k=int(os.environ.get("SCHOLAR_RAG_TOP_K", "10"),
        )
    )
