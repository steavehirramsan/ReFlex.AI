from __future__ import annotations

import numpy as np
import pytest

from reflex.config import EmbeddingConfig
from reflex.embeddings import build_embedder
from reflex.embeddings.hashing import HashingEmbedder


def test_shape_and_dtype() -> None:
    emb = HashingEmbedder(dim=64)
    out = emb.embed(["hello world", "another sentence"])
    assert out.shape == (2, 64)
    assert out.dtype == np.float32


def test_deterministic_across_instances() -> None:
    a = HashingEmbedder(dim=64).embed_one("the quick brown fox")
    b = HashingEmbedder(dim=64).embed_one("the quick brown fox")
    np.testing.assert_array_equal(a, b)


def test_normalized_vectors_are_unit_length() -> None:
    emb = HashingEmbedder(dim=64, normalize=True)
    v = emb.embed_one("some non-empty text here")
    assert pytest.approx(float(np.linalg.norm(v)), abs=1e-5) == 1.0


def test_similar_texts_more_similar_than_dissimilar() -> None:
    emb = HashingEmbedder(dim=256)
    base = emb.embed_one("my favorite programming language is python")
    near = emb.embed_one("my favorite programming language is python and rust")
    far = emb.embed_one("the weather in antarctica is extremely cold today")
    assert float(base @ near) > float(base @ far)


def test_empty_text_is_zero_vector() -> None:
    emb = HashingEmbedder(dim=32)
    v = emb.embed_one("")
    assert float(np.linalg.norm(v)) == 0.0


def test_invalid_dim_raises() -> None:
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)


def test_factory_builds_hashing() -> None:
    emb = build_embedder(EmbeddingConfig(provider="hashing", dim=48))
    assert emb.dim == 48
