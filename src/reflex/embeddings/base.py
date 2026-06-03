"""Embedding backend interface.

Embedders turn text into fixed-dimension vectors for similarity search. Like the LLM
layer, the runtime depends only on the abstract interface; the default is dependency-free
and deterministic so retrieval works offline.
"""

from __future__ import annotations

import abc

import numpy as np


class Embedder(abc.ABC):
    """Abstract text embedder."""

    #: Output vector dimensionality.
    dim: int

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts into a ``(len(texts), dim)`` float32 array."""
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text into a ``(dim,)`` float32 vector."""
        return np.asarray(self.embed([text])[0], dtype=np.float32)
