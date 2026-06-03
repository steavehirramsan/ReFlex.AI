from __future__ import annotations

import pytest

from reflex.config import MemoryConfig
from reflex.embeddings.hashing import HashingEmbedder
from reflex.memory.manager import MemoryManager, _default_summarizer
from reflex.types import Role


@pytest.fixture
def mm() -> MemoryManager:
    cfg = MemoryConfig(db_path=":memory:")
    m = MemoryManager(cfg, HashingEmbedder(dim=128))
    yield m
    m.close()


def test_record_event_hits_both_tiers(mm: MemoryManager) -> None:
    mm.record_event("s", "hello world", role=Role.USER)
    assert mm.episodic.count() == 1
    assert len(mm.short_term) == 1


def test_add_fact_is_retrievable(mm: MemoryManager) -> None:
    mm.add_fact("ReFlex runs on AMD Instinct accelerators", confidence=0.9)
    bundle = mm.retrieve("what hardware does reflex use")
    assert any("AMD" in h.content for h in bundle.hits)


def test_retrieve_fuses_and_ranks(mm: MemoryManager) -> None:
    mm.record_event("s", "the deployment uses kubernetes", role=Role.USER)
    mm.add_fact("ReFlex targets ROCm and AMD Instinct GPUs")
    bundle = mm.retrieve("what AMD Instinct GPUs does ReFlex target", total_k=5)
    assert not bundle.is_empty
    # Most relevant hit should mention GPUs/AMD.
    assert "AMD" in bundle.hits[0].content or "ROCm" in bundle.hits[0].content


def test_retrieve_respects_total_k(mm: MemoryManager) -> None:
    for i in range(10):
        mm.record_event("s", f"note number {i} about reflex memory tiers", role=Role.USER)
    bundle = mm.retrieve("reflex memory", total_k=3)
    assert len(bundle.hits) <= 3


def test_retrieve_min_score_filters(mm: MemoryManager) -> None:
    mm.config.retrieval.min_score = 0.99
    mm.record_event("s", "completely unrelated content about gardening", role=Role.USER)
    bundle = mm.retrieve("quantum chromodynamics in particle physics")
    assert bundle.is_empty


def test_compaction_moves_old_events_to_archive() -> None:
    cfg = MemoryConfig(db_path=":memory:")
    cfg.compaction.episodic_threshold = 5
    cfg.compaction.batch_size = 3
    cfg.compaction.keep_recent = 1
    m = MemoryManager(cfg, HashingEmbedder(dim=128))
    try:
        for i in range(6):
            m.record_event("s", f"event {i} with some content", role=Role.USER)
        before = m.episodic.count()
        archived = m.maybe_compact()
        assert archived == 3
        assert m.episodic.count() == before - 3
        assert m.archive.count() == 1
    finally:
        m.close()


def test_compaction_disabled_is_noop() -> None:
    cfg = MemoryConfig(db_path=":memory:")
    cfg.compaction.enabled = False
    cfg.compaction.episodic_threshold = 1
    m = MemoryManager(cfg, HashingEmbedder(dim=128))
    try:
        m.record_event("s", "one")
        m.record_event("s", "two")
        assert m.maybe_compact() == 0
        assert m.archive.count() == 0
    finally:
        m.close()


def test_archived_content_is_recallable() -> None:
    cfg = MemoryConfig(db_path=":memory:")
    cfg.compaction.episodic_threshold = 3
    cfg.compaction.batch_size = 3
    cfg.compaction.keep_recent = 0
    m = MemoryManager(cfg, HashingEmbedder(dim=128))
    try:
        for i in range(4):
            m.record_event("s", f"discussion about hierarchical memory compression {i}")
        m.maybe_compact()
        bundle = m.retrieve("hierarchical memory compression")
        assert any(h.tier.value == "archive" for h in bundle.hits)
    finally:
        m.close()


def test_default_summarizer() -> None:
    out = _default_summarizer(["alpha", "beta", "gamma"])
    assert "3 records" in out and "alpha" in out


def test_stats_shape(mm: MemoryManager) -> None:
    mm.record_event("s", "x")
    stats = mm.stats()
    assert set(stats) == {"short_term", "episodic", "semantic", "archive", "working_tokens"}
