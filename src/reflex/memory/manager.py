"""The memory manager: the single front door to the tiered memory subsystem.

It owns the embedder, the three durable stores (episodic, semantic, archive), and the two
volatile tiers (short-term buffer, working memory). It implements the cross-tier behaviours
the README describes:

* **recording** events into the volatile buffer *and* the durable episodic log;
* **unified retrieval** that searches every durable tier, fuses scores with a recency
  prior, deduplicates, and returns a single ranked :class:`~reflex.types.RetrievalBundle`;
* **autonomous compaction** that summarises the oldest durable records into the cold
  archive once a tier crosses its size threshold, keeping resident memory bounded.

The manager is deliberately LLM-free. Summarisation during compaction is delegated to an
injectable callable so the durable layer stays fully testable offline; the agent wires an
LLM-backed summariser in production.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from ..config import MemoryConfig
from ..embeddings.base import Embedder
from ..logging import get_logger
from ..types import (
    ArchiveRecord,
    Event,
    Fact,
    MemoryTier,
    RetrievalBundle,
    Role,
    new_id,
    now,
)
from .archive import ArchiveStore
from .db import Database
from .episodic import EpisodicStore
from .semantic import SemanticStore
from .short_term import ShortTermBuffer, WorkingMemory, estimate_tokens
from .vector_index import VectorIndex, build_vector_index

log = get_logger(__name__)

Summarizer = Callable[[list[str]], str]


def _default_summarizer(texts: list[str]) -> str:
    """Heuristic, offline summariser: a truncated concatenation. Deterministic."""
    joined = " | ".join(t.strip().replace("\n", " ") for t in texts if t.strip())
    return f"Summary of {len(texts)} records: {joined[:800]}"


class MemoryManager:
    """Coordinates all memory tiers behind one interface."""

    def __init__(
        self,
        config: MemoryConfig,
        embedder: Embedder,
        *,
        db: Database | None = None,
        summarizer: Summarizer | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.db = db or Database(config.db_path)
        self._summarize = summarizer or _default_summarizer

        def mk_index() -> VectorIndex:
            return build_vector_index(embedder.dim, config.vector.backend, config.vector.metric)

        self.episodic = EpisodicStore(self.db, embedder, mk_index())
        self.semantic = SemanticStore(self.db, embedder, mk_index())
        self.archive = ArchiveStore(self.db, embedder, mk_index())
        self.short_term = ShortTermBuffer(config.short_term_capacity)
        self.working = WorkingMemory(config.working_token_budget)

    # -- recording ---------------------------------------------------------

    def record_event(
        self,
        session_id: str,
        content: str,
        *,
        role: Role = Role.USER,
        kind: str = "message",
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> Event:
        """Record an event into the short-term buffer and the durable episodic log."""
        event = Event(
            session_id=session_id,
            content=content,
            role=role,
            kind=kind,
            importance=importance,
            metadata=metadata or {},
        )
        self.short_term.add(event)  # buffer eviction is non-destructive: episodic is durable
        self.episodic.add(event)
        return event

    def add_fact(
        self,
        statement: str,
        *,
        confidence: float = 0.7,
        source_event_ids: list[str] | None = None,
        subject: str | None = None,
        predicate: str | None = None,
        object_: str | None = None,
    ) -> Fact:
        """Distil a new semantic fact into the durable fact base."""
        fact = Fact(
            statement=statement,
            confidence=confidence,
            source_event_ids=source_event_ids or [],
            subject=subject,
            predicate=predicate,
            object=object_,
        )
        return self.semantic.upsert(fact)

    # -- retrieval ---------------------------------------------------------

    def retrieve(self, query: str, *, total_k: int | None = None) -> RetrievalBundle:
        """Search every durable tier and fuse the results into one ranked bundle.

        Final score = (1 - w) * similarity + w * recency, where recency decays
        exponentially with a configurable half-life. Hits below ``min_score`` similarity are
        dropped before fusion so a recency boost can never resurrect an irrelevant memory.
        """
        rc = self.config.retrieval
        total_k = total_k or rc.total_k
        qvec = self.embedder.embed_one(query)

        raw = [
            *self.episodic.search(qvec, rc.episodic_k),
            *self.semantic.search(qvec, rc.semantic_k),
            *self.archive.search(qvec, rc.archive_k),
        ]
        t = now()
        scored = []
        seen: set[str] = set()
        for hit in raw:
            if hit.score < rc.min_score or hit.record_id in seen:
                continue
            seen.add(hit.record_id)
            recency = math.exp(-max(0.0, t - hit.ts) / rc.recency_half_life_s)
            fused = (1 - rc.recency_weight) * hit.score + rc.recency_weight * recency
            scored.append((fused, hit))

        scored.sort(key=lambda x: x[0], reverse=True)
        bundle = RetrievalBundle(query=query)
        for fused, hit in scored[:total_k]:
            bundle.hits.append(hit.model_copy(update={"score": round(fused, 6)}))
        return bundle

    # -- compaction --------------------------------------------------------

    def maybe_compact(self) -> int:
        """Compact overflowing durable tiers into the archive. Returns records archived."""
        if not self.config.compaction.enabled:
            return 0
        archived = 0
        cc = self.config.compaction
        if self.episodic.count() > cc.episodic_threshold:
            archived += self._compact_episodic()
        return archived

    def _compact_episodic(self) -> int:
        cc = self.config.compaction
        batch = self.episodic.oldest(cc.batch_size, exclude_recent=cc.keep_recent)
        if not batch:
            return 0
        summary = self._summarize([e.content for e in batch])
        record = ArchiveRecord(
            summary=summary,
            origin_tier=MemoryTier.EPISODIC,
            span_start=min(e.ts for e in batch),
            span_end=max(e.ts for e in batch),
            source_ids=[e.id for e in batch],
            token_estimate=estimate_tokens(summary),
        )
        self.archive.add(record)
        self.episodic.delete([e.id for e in batch])
        log.info("Compacted %d episodic events into archive record %s", len(batch), record.id)
        return len(batch)

    # -- introspection -----------------------------------------------------

    def stats(self) -> dict[str, int]:
        return {
            "short_term": len(self.short_term),
            "episodic": self.episodic.count(),
            "semantic": self.semantic.count(),
            "archive": self.archive.count(),
            "working_tokens": self.working.tokens,
        }

    def new_session_id(self) -> str:
        return new_id("sess")

    def close(self) -> None:
        self.db.close()
