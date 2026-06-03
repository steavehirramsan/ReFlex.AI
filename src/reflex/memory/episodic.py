"""Episodic store — the durable, time-stamped log of *what happened*.

Events are append-only and persisted in SQLite. Each event is also embedded and inserted
into a vector index so the orchestrator can retrieve relevant history by similarity. On
construction the index is rebuilt from any persisted embeddings, so a process restart
recovers the full searchable history.
"""

from __future__ import annotations

import sqlite3

import numpy as np

from ..embeddings.base import Embedder
from ..types import Event, MemoryHit, MemoryTier, Role
from .db import Database, dumps_embedding, dumps_json, loads_embedding, loads_json
from .vector_index import VectorIndex


class EpisodicStore:
    """Durable, searchable event history."""

    def __init__(self, db: Database, embedder: Embedder, index: VectorIndex) -> None:
        self._db = db
        self._embedder = embedder
        self._index = index
        self._load_index()

    def _load_index(self) -> None:
        rows = self._db.query("SELECT id, embedding FROM events WHERE embedding IS NOT NULL")
        ids, vecs = [], []
        for row in rows:
            emb = loads_embedding(row["embedding"])
            if emb is not None:
                ids.append(row["id"])
                vecs.append(emb)
        if ids:
            self._index.add(ids, np.array(vecs, dtype=np.float32))

    # -- writes ------------------------------------------------------------

    def add(self, event: Event) -> Event:
        """Persist an event (embedding it if not already embedded) and index it."""
        if event.embedding is None:
            event = event.model_copy(
                update={"embedding": self._embedder.embed_one(event.content).tolist()}
            )
        self._db.execute(
            """INSERT OR REPLACE INTO events
               (id, session_id, ts, kind, role, content, importance, metadata, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.id,
                event.session_id,
                event.ts,
                event.kind,
                event.role.value,
                event.content,
                event.importance,
                dumps_json(event.metadata),
                dumps_embedding(event.embedding),
            ),
        )
        if event.embedding is not None:
            self._index.add([event.id], np.array([event.embedding], dtype=np.float32))
        return event

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        self._db.execute(f"DELETE FROM events WHERE id IN ({placeholders})", ids)
        self._index.remove(ids)

    # -- reads -------------------------------------------------------------

    def get(self, event_id: str) -> Event | None:
        row = self._db.query_one("SELECT * FROM events WHERE id = ?", (event_id,))
        return _row_to_event(row) if row else None

    def recent(self, session_id: str | None = None, n: int = 20) -> list[Event]:
        if session_id is None:
            rows = self._db.query("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (n,))
        else:
            rows = self._db.query(
                "SELECT * FROM events WHERE session_id = ? ORDER BY ts DESC LIMIT ?",
                (session_id, n),
            )
        return [_row_to_event(r) for r in reversed(rows)]

    def oldest(self, n: int, *, exclude_recent: int = 0) -> list[Event]:
        """Return the ``n`` oldest events, optionally sparing the most recent ones."""
        total = self.count()
        limit = max(0, min(n, total - exclude_recent))
        if limit <= 0:
            return []
        rows = self._db.query("SELECT * FROM events ORDER BY ts ASC LIMIT ?", (limit,))
        return [_row_to_event(r) for r in rows]

    def search(self, query_vec: np.ndarray, k: int) -> list[MemoryHit]:
        hits = self._index.search(query_vec, k)
        if not hits:
            return []
        by_id = {h[0]: h[1] for h in hits}
        placeholders = ",".join("?" * len(by_id))
        rows = self._db.query(
            f"SELECT * FROM events WHERE id IN ({placeholders})",
            list(by_id),
        )
        out = [
            MemoryHit(
                record_id=r["id"],
                tier=MemoryTier.EPISODIC,
                content=r["content"],
                score=by_id[r["id"]],
                ts=r["ts"],
                metadata={"kind": r["kind"], "role": r["role"]},
            )
            for r in rows
        ]
        out.sort(key=lambda h: h.score, reverse=True)
        return out

    def count(self) -> int:
        return self._db.count("events")


def _row_to_event(row: sqlite3.Row) -> Event:
    return Event(
        id=row["id"],
        session_id=row["session_id"],
        ts=row["ts"],
        kind=row["kind"],
        role=Role(row["role"]),
        content=row["content"],
        importance=row["importance"],
        metadata=loads_json(row["metadata"], {}),
        embedding=loads_embedding(row["embedding"]),
    )
