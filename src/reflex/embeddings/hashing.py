"""Deterministic, dependency-free hashing embedder.

Uses the classic *feature hashing* trick: lowercase word unigrams and bigrams are hashed
(with a stable, salt-free hash) into a fixed number of buckets with signed accumulation,
then L2-normalised. This gives a deterministic vector whose cosine similarity tracks
lexical overlap — enough for retrieval to behave sensibly offline and in tests, with zero
model downloads. For semantic-quality embeddings on real deployments, switch to the
``sentence_transformers`` backend (``reflex-memory[embeddings]``).
"""

from __future__ import annotations

import hashlib
import re
from itertools import pairwise

import numpy as np

from .base import Embedder

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stable_hash(token: str) -> int:
    """Salt-free, process-independent hash (Python's ``hash`` is randomised per run)."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


class HashingEmbedder(Embedder):
    """Feature-hashing embedder with unigram + bigram features."""

    def __init__(self, dim: int = 256, *, normalize: bool = True) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self._normalize = normalize

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            self._embed_into(text, out[i])
        if self._normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            np.divide(out, norms, out=out, where=norms > 0)
        return out

    def _embed_into(self, text: str, row: np.ndarray) -> None:
        tokens = _TOKEN_RE.findall(text.lower())
        features = list(tokens)
        features.extend(f"{a}_{b}" for a, b in pairwise(tokens))
        for feat in features:
            h = _stable_hash(feat)
            bucket = h % self.dim
            sign = 1.0 if (h >> 1) & 1 else -1.0
            row[bucket] += sign
