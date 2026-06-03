"""Approximate/exact nearest-neighbour indexes used by the memory tiers.

The default :class:`NumpyVectorIndex` is exact, dependency-free, and fine for research-scale
stores (tens of thousands of vectors). For larger deployments install ``reflexai[faiss]``
and select ``vector.backend: faiss`` — :class:`FaissVectorIndex` keeps the same interface.
"""

from __future__ import annotations

import abc
from typing import Literal

import numpy as np

Metric = Literal["cosine", "ip", "l2"]


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    out = vecs.astype(np.float32, copy=True)
    np.divide(out, norms, out=out, where=norms > 0)
    return out


class VectorIndex(abc.ABC):
    """Abstract id-keyed vector store with top-k similarity search."""

    dim: int
    metric: Metric

    @abc.abstractmethod
    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...

    @abc.abstractmethod
    def remove(self, ids: list[str]) -> None: ...

    @abc.abstractmethod
    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(id, similarity)`` pairs, highest similarity first.

        Similarity is in ``[-1, 1]`` for cosine, raw dot for ``ip``, and ``-distance`` for
        ``l2`` (so that "larger is better" holds for every metric).
        """

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def __contains__(self, id_: str) -> bool: ...


class NumpyVectorIndex(VectorIndex):
    """Exact brute-force index backed by a single contiguous ``numpy`` matrix."""

    def __init__(self, dim: int, metric: Metric = "cosine") -> None:
        self.dim = dim
        self.metric = metric
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._matrix = np.zeros((0, dim), dtype=np.float32)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32).reshape(len(ids), self.dim)
        if self.metric == "cosine":
            vectors = _normalize(vectors)
        # Upsert: overwrite rows for ids we already hold, append the rest.
        fresh_ids: list[str] = []
        fresh_rows: list[np.ndarray] = []
        for id_, vec in zip(ids, vectors, strict=True):
            if id_ in self._pos:
                self._matrix[self._pos[id_]] = vec
            else:
                fresh_ids.append(id_)
                fresh_rows.append(vec)
        if fresh_ids:
            start = len(self._ids)
            self._ids.extend(fresh_ids)
            for offset, id_ in enumerate(fresh_ids):
                self._pos[id_] = start + offset
            self._matrix = np.vstack([self._matrix, np.array(fresh_rows, dtype=np.float32)])

    def remove(self, ids: list[str]) -> None:
        drop = {i for i in ids if i in self._pos}
        if not drop:
            return
        keep_mask = np.array([i not in drop for i in self._ids], dtype=bool)
        self._matrix = self._matrix[keep_mask]
        self._ids = [i for i in self._ids if i not in drop]
        self._pos = {id_: idx for idx, id_ in enumerate(self._ids)}

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if k <= 0 or len(self._ids) == 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(self.dim)
        if self.metric == "cosine":
            qn = np.linalg.norm(q)
            if qn > 0:
                q = q / qn
            scores = self._matrix @ q
        elif self.metric == "ip":
            scores = self._matrix @ q
        else:  # l2 -> negative distance
            diff = self._matrix - q
            scores = -np.linalg.norm(diff, axis=1)
        k = min(k, len(self._ids))
        # argpartition for the top-k, then sort just those k.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[i], float(scores[i])) for i in top]

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, id_: str) -> bool:
        return id_ in self._pos


class FaissVectorIndex(VectorIndex):
    """FAISS-backed index (optional, ``reflexai[faiss]``) for large stores."""

    def __init__(self, dim: int, metric: Metric = "cosine") -> None:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - only without the extra
            from ..errors import BackendError

            raise BackendError(
                'faiss is not installed. Install with `pip install "reflexai[faiss]"`.'
            ) from exc
        self._faiss = faiss
        self.dim = dim
        self.metric = metric
        base = faiss.IndexFlatIP(dim) if metric in ("cosine", "ip") else faiss.IndexFlatL2(dim)
        self._index = faiss.IndexIDMap2(base)
        self._ids: list[str] = []
        self._id_to_int: dict[str, int] = {}
        self._int_to_id: dict[int, str] = {}
        self._counter = 0

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if not ids:
            return
        vectors = np.asarray(vectors, dtype=np.float32).reshape(len(ids), self.dim)
        if self.metric == "cosine":
            vectors = _normalize(vectors)
        existing = [i for i in ids if i in self._id_to_int]
        if existing:
            self.remove(existing)
        int_ids = []
        for id_ in ids:
            self._counter += 1
            self._id_to_int[id_] = self._counter
            self._int_to_id[self._counter] = id_
            self._ids.append(id_)
            int_ids.append(self._counter)
        self._index.add_with_ids(vectors, np.array(int_ids, dtype=np.int64))

    def remove(self, ids: list[str]) -> None:
        int_ids = [self._id_to_int[i] for i in ids if i in self._id_to_int]
        if not int_ids:
            return
        self._index.remove_ids(np.array(int_ids, dtype=np.int64))
        for i in ids:
            iid = self._id_to_int.pop(i, None)
            if iid is not None:
                self._int_to_id.pop(iid, None)
        drop = set(ids)
        self._ids = [i for i in self._ids if i not in drop]

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if k <= 0 or len(self._ids) == 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(1, self.dim)
        if self.metric == "cosine":
            q = _normalize(q)
        k = min(k, len(self._ids))
        scores, idx = self._index.search(q, k)
        out: list[tuple[str, float]] = []
        for score, iid in zip(scores[0], idx[0], strict=True):
            if iid == -1:
                continue
            id_ = self._int_to_id.get(int(iid))
            if id_ is None:
                continue
            sim = float(score) if self.metric in ("cosine", "ip") else -float(score)
            out.append((id_, sim))
        return out

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, id_: str) -> bool:
        return id_ in self._id_to_int


def build_vector_index(dim: int, backend: str, metric: Metric) -> VectorIndex:
    """Construct the configured vector index."""
    if backend == "numpy":
        return NumpyVectorIndex(dim, metric)
    if backend == "faiss":
        return FaissVectorIndex(dim, metric)
    raise ValueError(f"Unknown vector backend: {backend!r}")  # pragma: no cover
