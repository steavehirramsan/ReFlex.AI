from __future__ import annotations

import numpy as np
import pytest

from reflex.memory.vector_index import NumpyVectorIndex, build_vector_index


@pytest.fixture
def index() -> NumpyVectorIndex:
    return NumpyVectorIndex(dim=3, metric="cosine")


def test_add_and_search_returns_nearest(index: NumpyVectorIndex) -> None:
    index.add(["a", "b", "c"], np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32))
    results = index.search(np.array([1, 0, 0], dtype=np.float32), k=2)
    assert results[0][0] == "a"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
    assert len(results) == 2


def test_len_and_contains(index: NumpyVectorIndex) -> None:
    index.add(["a"], np.array([[1, 0, 0]], dtype=np.float32))
    assert len(index) == 1
    assert "a" in index
    assert "z" not in index


def test_upsert_overwrites(index: NumpyVectorIndex) -> None:
    index.add(["a"], np.array([[1, 0, 0]], dtype=np.float32))
    index.add(["a"], np.array([[0, 1, 0]], dtype=np.float32))
    assert len(index) == 1
    best = index.search(np.array([0, 1, 0], dtype=np.float32), k=1)
    assert best[0][0] == "a" and best[0][1] == pytest.approx(1.0, abs=1e-5)


def test_remove(index: NumpyVectorIndex) -> None:
    index.add(["a", "b"], np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    index.remove(["a"])
    assert "a" not in index and len(index) == 1
    results = index.search(np.array([1, 0, 0], dtype=np.float32), k=5)
    assert all(r[0] != "a" for r in results)


def test_search_empty_index_returns_empty(index: NumpyVectorIndex) -> None:
    assert index.search(np.array([1, 0, 0], dtype=np.float32), k=3) == []


def test_search_k_zero(index: NumpyVectorIndex) -> None:
    index.add(["a"], np.array([[1, 0, 0]], dtype=np.float32))
    assert index.search(np.array([1, 0, 0], dtype=np.float32), k=0) == []


def test_ip_metric_ranks_by_dot() -> None:
    idx = NumpyVectorIndex(dim=2, metric="ip")
    idx.add(["small", "big"], np.array([[1, 0], [5, 0]], dtype=np.float32))
    results = idx.search(np.array([1, 0], dtype=np.float32), k=2)
    assert results[0][0] == "big"  # larger dot product


def test_l2_metric_ranks_by_distance() -> None:
    idx = NumpyVectorIndex(dim=2, metric="l2")
    idx.add(["near", "far"], np.array([[1, 0], [9, 9]], dtype=np.float32))
    results = idx.search(np.array([1, 0], dtype=np.float32), k=2)
    assert results[0][0] == "near"


def test_factory() -> None:
    idx = build_vector_index(4, "numpy", "cosine")
    assert isinstance(idx, NumpyVectorIndex) and idx.dim == 4


def test_factory_unknown_backend_raises() -> None:
    with pytest.raises(ValueError):
        build_vector_index(4, "nope", "cosine")
