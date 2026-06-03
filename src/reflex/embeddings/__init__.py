"""Embedding backends and the factory that builds one from config."""

from __future__ import annotations

from ..config import EmbeddingConfig
from .base import Embedder
from .hashing import HashingEmbedder

__all__ = ["Embedder", "HashingEmbedder", "build_embedder"]


def build_embedder(config: EmbeddingConfig) -> Embedder:
    """Instantiate the configured embedding backend."""
    if config.provider == "hashing":
        return HashingEmbedder(dim=config.dim, normalize=config.normalize)
    if config.provider == "sentence_transformers":
        from .sentence_transformers import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(model=config.model, normalize=config.normalize)
    raise ValueError(f"Unknown embedding provider: {config.provider!r}")  # pragma: no cover
