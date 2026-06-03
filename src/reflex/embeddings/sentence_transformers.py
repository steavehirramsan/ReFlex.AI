"""Optional sentence-transformers embedding backend.

Requires the ``embeddings`` extra::

    pip install "reflex-memory[embeddings]"

On AMD Instinct, install the ROCm PyTorch build first so the model runs on-GPU.
"""

from __future__ import annotations

import numpy as np

from ..errors import BackendError
from .base import Embedder


class SentenceTransformerEmbedder(Embedder):
    """Wraps a ``sentence-transformers`` model behind the :class:`Embedder` interface."""

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise BackendError(
                "sentence-transformers is not installed. Install with "
                '`pip install "reflex-memory[embeddings]"`.'
            ) from exc
        self._model = SentenceTransformer(model, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self._normalize = normalize

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)
