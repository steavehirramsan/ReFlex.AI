"""Compressed archive — the cold tier of summarised long-tail history.

When the durable tiers grow past their thresholds, the memory manager compacts the oldest
records into :class:`~reflex.types.ArchiveRecord` summaries here. The archive stays
searchable by embedding so months-old context can still be recalled without holding every
raw event resident.
"""

from __future__ import annotations

import sqlite3

import numpy as np

from ..embeddings.base import Embedder
from ..types import ArchiveRecord, MemoryHit, MemoryTier
from .db import Database, dumps_embedding, dumps_json, loads_embedding, loads_json
from .vector_index import VectorIndex


class ArchiveStore:
    """Durable, searchable store of compressed summaries."""

    def __init__(self, db: Database, embedder: Embedder, index: VectorIndex) -> None:
        self._db = db
        self._embedder = embedder
        self._index = index
        self._load_index()

    def _load_index(self) -> None:
        rows = self._db.query("SELECT id, embedding FROM archive WHERE embedding IS NOT NULL")
        ids, vecs = [], []
        for row in rows:
            emb = loads_embedding(row["embedding"])
            if emb is not None:
                ids.append(row["id"])
                vecs.append(emb)
        if ids:
            self._index.add(ids, np.array(vecs, dtype=np.float32))

    def add(self, record: ArchiveRecord) -> ArchiveRecord:
        if record.embedding is None:
            record = record.model_copy(
                update={"embedding": self._embedder.embed_one(record.summary).tolist()}
            )
        self._db.execute(
            """INSERT OR REPLACE INTO archive
               (id, summary, origin_tier, span_start, span_end,
                source_ids, token_estimate, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id,
                record.summary,
                record.origin_tier.value,
                record.span_start,
                record.span_end,
                dumps_json(record.source_ids),
                record.token_estimate,
                dumps_embedding(record.embedding),
            ),
        )
        if record.embedding is not None:
            self._index.add([record.id], np.array([record.embedding], dtype=np.float32))
        return record

    def search(self, query_vec: np.ndarray, k: int) -> list[MemoryHit]:
        hits = self._index.search(query_vec, k)
        if not hits:
            return []
        by_id = {h[0]: h[1] for h in hits}
        placeholders = ",".join("?" * len(by_id))
        rows = self._db.query(
            f"SELECT * FROM archive WHERE id IN ({placeholders})",
            list(by_id),
        )
        out = [
            MemoryHit(
                record_id=r["id"],
                tier=MemoryTier.ARCHIVE,
                content=r["summary"],
                score=by_id[r["id"]],
                ts=r["span_end"],
                metadata={"origin_tier": r["origin_tier"]},
            )
            for r in rows
        ]
        out.sort(key=lambda h: h.score, reverse=True)
        return out

    def get(self, record_id: str) -> ArchiveRecord | None:
        row = self._db.query_one("SELECT * FROM archive WHERE id = ?", (record_id,))
        return _row_to_archive(row) if row else None

    def count(self) -> int:
        return self._db.count("archive")


def _row_to_archive(row: sqlite3.Row) -> ArchiveRecord:
    return ArchiveRecord(
        id=row["id"],
        summary=row["summary"],
        origin_tier=MemoryTier(row["origin_tier"]),
        span_start=row["span_start"],
        span_end=row["span_end"],
        source_ids=loads_json(row["source_ids"], []),
        token_estimate=row["token_estimate"],
        embedding=loads_embedding(row["embedding"]),
    )
