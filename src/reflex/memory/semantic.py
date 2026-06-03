"""Semantic store — the durable set of distilled facts: *what is true*.

Unlike episodic events, facts are mutable: a fact can be invalidated and superseded when
the integrity layer detects that an established belief no longer holds. Only ``valid`` facts
are searchable, so retrieval never surfaces a belief the system has retracted.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from ..embeddings.base import Embedder
from ..types import Fact, MemoryHit, MemoryTier, now
from .db import Database, dumps_embedding, dumps_json, loads_embedding, loads_json
from .vector_index import VectorIndex


class SemanticStore:
    """Durable, searchable fact base with supersession semantics."""

    def __init__(self, db: Database, embedder: Embedder, index: VectorIndex) -> None:
        self._db = db
        self._embedder = embedder
        self._index = index
        self._load_index()

    def _load_index(self) -> None:
        rows = self._db.query(
            "SELECT id, embedding FROM facts WHERE valid = 1 AND embedding IS NOT NULL"
        )
        ids, vecs = [], []
        for row in rows:
            emb = loads_embedding(row["embedding"])
            if emb is not None:
                ids.append(row["id"])
                vecs.append(emb)
        if ids:
            self._index.add(ids, np.array(vecs, dtype=np.float32))

    # -- writes ------------------------------------------------------------

    def upsert(self, fact: Fact) -> Fact:
        """Insert or update a fact, embedding it if needed and (de)indexing by validity."""
        if fact.embedding is None:
            fact = fact.model_copy(
                update={"embedding": self._embedder.embed_one(fact.statement).tolist()}
            )
        self._db.execute(
            """INSERT OR REPLACE INTO facts
               (id, statement, subject, predicate, object, confidence, source_event_ids,
                created_ts, updated_ts, valid, superseded_by, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fact.id,
                fact.statement,
                fact.subject,
                fact.predicate,
                fact.object,
                fact.confidence,
                dumps_json(fact.source_event_ids),
                fact.created_ts,
                fact.updated_ts,
                int(fact.valid),
                fact.superseded_by,
                dumps_embedding(fact.embedding),
            ),
        )
        if fact.valid and fact.embedding is not None:
            self._index.add([fact.id], np.array([fact.embedding], dtype=np.float32))
        else:
            self._index.remove([fact.id])
        return fact

    def invalidate(self, fact_id: str, *, superseded_by: str | None = None) -> None:
        """Mark a fact invalid (and optionally point at the fact that replaced it)."""
        self._db.execute(
            "UPDATE facts SET valid = 0, superseded_by = ?, updated_ts = ? WHERE id = ?",
            (superseded_by, now(), fact_id),
        )
        self._index.remove([fact_id])

    # -- reads -------------------------------------------------------------

    def get(self, fact_id: str) -> Fact | None:
        row = self._db.query_one("SELECT * FROM facts WHERE id = ?", (fact_id,))
        return _row_to_fact(row) if row else None

    def search(self, query_vec: np.ndarray, k: int) -> list[MemoryHit]:
        hits = self._index.search(query_vec, k)
        if not hits:
            return []
        by_id = {h[0]: h[1] for h in hits}
        placeholders = ",".join("?" * len(by_id))
        rows = self._db.query(
            f"SELECT * FROM facts WHERE id IN ({placeholders}) AND valid = 1",
            list(by_id),
        )
        out = [
            MemoryHit(
                record_id=r["id"],
                tier=MemoryTier.SEMANTIC,
                content=r["statement"],
                score=by_id[r["id"]],
                ts=r["updated_ts"],
                metadata={"confidence": r["confidence"]},
            )
            for r in rows
        ]
        out.sort(key=lambda h: h.score, reverse=True)
        return out

    def all_valid(self, limit: int | None = None) -> list[Fact]:
        sql = "SELECT * FROM facts WHERE valid = 1 ORDER BY updated_ts DESC"
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        return [_row_to_fact(r) for r in self._db.query(sql, params)]

    def count(self, *, valid_only: bool = True) -> int:
        return self._db.count("facts", "valid = 1" if valid_only else "")


def _row_to_fact(row: sqlite3.Row) -> Fact:
    return Fact(
        id=row["id"],
        statement=row["statement"],
        subject=row["subject"],
        predicate=row["predicate"],
        object=row["object"],
        confidence=row["confidence"],
        source_event_ids=loads_json(row["source_event_ids"], []),
        created_ts=row["created_ts"],
        updated_ts=row["updated_ts"],
        valid=bool(row["valid"]),
        superseded_by=row["superseded_by"],
        embedding=loads_embedding(row["embedding"]),
    )
