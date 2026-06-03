from __future__ import annotations

import numpy as np

from reflex.embeddings.hashing import HashingEmbedder
from reflex.memory.archive import ArchiveStore
from reflex.memory.db import Database, dumps_embedding, loads_embedding
from reflex.memory.episodic import EpisodicStore
from reflex.memory.semantic import SemanticStore
from reflex.memory.vector_index import NumpyVectorIndex
from reflex.types import ArchiveRecord, Event, Fact, MemoryTier, Role


def _idx(dim: int = 128) -> NumpyVectorIndex:
    return NumpyVectorIndex(dim, "cosine")


def test_embedding_roundtrip() -> None:
    v = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    assert loads_embedding(dumps_embedding(v)) == [0.1, -0.2, 0.3]
    assert loads_embedding(None) is None
    assert dumps_embedding(None) is None


def test_database_schema_version() -> None:
    db = Database(":memory:")
    (version,) = db.query_one("PRAGMA user_version")
    assert version >= 1
    db.close()


def test_episodic_add_get_search() -> None:
    db = Database(":memory:")
    emb = HashingEmbedder(dim=128)
    store = EpisodicStore(db, emb, _idx())
    store.add(Event(session_id="s", content="python is my favorite language", role=Role.USER))
    store.add(Event(session_id="s", content="the cat sat on the mat", role=Role.USER))

    assert store.count() == 2
    hits = store.search(emb.embed_one("which programming language do I like"), k=1)
    assert hits and hits[0].tier == MemoryTier.EPISODIC
    assert "python" in hits[0].content
    db.close()


def test_episodic_recent_order_and_filter() -> None:
    db = Database(":memory:")
    emb = HashingEmbedder(dim=128)
    store = EpisodicStore(db, emb, _idx())
    for i in range(3):
        store.add(Event(session_id="s", ts=float(i), content=f"msg{i}"))
    store.add(Event(session_id="other", ts=10.0, content="elsewhere"))
    recent = store.recent("s", n=2)
    assert [e.content for e in recent] == ["msg1", "msg2"]  # oldest-first, session-scoped
    db.close()


def test_episodic_persists_across_reopen(tmp_path) -> None:
    path = tmp_path / "mem.db"
    emb = HashingEmbedder(dim=128)
    db = Database(path)
    store = EpisodicStore(db, emb, _idx())
    store.add(Event(session_id="s", content="durable knowledge about reflex memory"))
    db.close()

    db2 = Database(path)
    store2 = EpisodicStore(db2, emb, _idx())  # rebuilds index from persisted embeddings
    assert store2.count() == 1
    hits = store2.search(emb.embed_one("reflex memory"), k=1)
    assert hits and "durable" in hits[0].content
    db2.close()


def test_episodic_delete_removes_from_index() -> None:
    db = Database(":memory:")
    emb = HashingEmbedder(dim=128)
    store = EpisodicStore(db, emb, _idx())
    ev = store.add(Event(session_id="s", content="to be deleted soon"))
    store.delete([ev.id])
    assert store.count() == 0
    assert store.search(emb.embed_one("deleted"), k=5) == []
    db.close()


def test_semantic_upsert_search_and_invalidate() -> None:
    db = Database(":memory:")
    emb = HashingEmbedder(dim=128)
    store = SemanticStore(db, emb, _idx())
    fact = store.upsert(Fact(statement="The capital of France is Paris"))
    hits = store.search(emb.embed_one("capital of France"), k=1)
    assert hits and "Paris" in hits[0].content

    store.invalidate(fact.id)
    assert store.count(valid_only=True) == 0
    assert store.search(emb.embed_one("capital of France"), k=1) == []
    db.close()


def test_semantic_supersession_pointer() -> None:
    db = Database(":memory:")
    emb = HashingEmbedder(dim=128)
    store = SemanticStore(db, emb, _idx())
    old = store.upsert(Fact(statement="The project lead is Alice"))
    new = store.upsert(Fact(statement="The project lead is Bob"))
    store.invalidate(old.id, superseded_by=new.id)
    reloaded = store.get(old.id)
    assert reloaded is not None and reloaded.superseded_by == new.id and not reloaded.valid
    db.close()


def test_archive_add_and_search() -> None:
    db = Database(":memory:")
    emb = HashingEmbedder(dim=128)
    store = ArchiveStore(db, emb, _idx())
    store.add(ArchiveRecord(summary="A long discussion about distributed training on ROCm"))
    assert store.count() == 1
    hits = store.search(emb.embed_one("ROCm training"), k=1)
    assert hits and hits[0].tier == MemoryTier.ARCHIVE
    db.close()
