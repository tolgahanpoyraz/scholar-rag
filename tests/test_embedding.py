from __future__ import annotations

import math

import pytest

from tests.fakes import FakeEmbedder

def test_embed_texts_shape():
    emb = FakeEmbedder(dimension=8)
    vecs = emb.embed_texts(["a", "b", "c"])
    assert len(vecs) == 3
    assert all(len(v) == 8 for v in vecs)

def test_embed_query_delegates():
    emb = FakeEmbedder(dimension=8)
    assert emb.embed_query("hello") == emb.embed_texts(["hello"])[0]

def test_deterministic():
    emb = FakeEmbedder()
    assert emb.embed_query("same") == emb.embed_query("same")

def test_vectors_normalized():
    emb = FakeEmbedder(dimension=16)
    v = emb.embed_query("anything")
    assert abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-9

@pytest.mark.slow
def test_real_embedder_smoke():
    """Loads MiniLM. Run with: uv run pytest -m slow"""
    from scholar_rag.embedding.sentence_transformer import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder()
    assert emb.dimension == 384            # MiniLM-L6-v2 is 384-dimensional
    vecs = emb.embed_texts(["graph theory", "a cooking recipe"])
    assert len(vecs) == 2 and len(vecs[0]) == 384
    assert abs(math.sqrt(sum(x * x for x in vecs[0])) - 1.0) < 1e-5