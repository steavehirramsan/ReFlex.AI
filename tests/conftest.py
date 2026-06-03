"""Shared pytest fixtures.

Every fixture uses in-memory SQLite and the deterministic offline backends so the suite
runs fast, hermetically, and without a GPU, network, or API key.
"""

from __future__ import annotations

import pytest

from reflex.config import ReflexConfig
from reflex.embeddings.hashing import HashingEmbedder
from reflex.memory.manager import MemoryManager


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=128)


@pytest.fixture
def config() -> ReflexConfig:
    """A default config wired to an in-memory database with a small embedding dim."""
    return ReflexConfig.load(
        env=False,
        overrides={
            "memory": {"db_path": ":memory:", "vector": {"backend": "numpy"}},
            "embeddings": {"dim": 128},
            "logging": {"rich": False, "level": "WARNING"},
        },
    )


@pytest.fixture
def memory(config: ReflexConfig, embedder: HashingEmbedder) -> MemoryManager:
    mm = MemoryManager(config.memory, embedder)
    yield mm
    mm.close()
