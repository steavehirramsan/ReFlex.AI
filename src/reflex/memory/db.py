"""SQLite-backed storage for the durable memory tiers.

A single database file holds the episodic, semantic, and archive tables. The schema is
created idempotently on open and versioned via ``PRAGMA user_version`` so future migrations
have a hook. Access is guarded by a re-entrant lock and the connection is opened with
``check_same_thread=False`` so the synchronous stores can be called safely from the async
orchestrator's thread pool.

Postgres is intentionally out of scope for the default build; the ``Database`` surface is
small (``execute`` / ``query`` / ``executemany``) precisely so a Postgres adapter can be
dropped in behind the same store classes later.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    ts            REAL NOT NULL,
    kind          TEXT NOT NULL,
    role          TEXT NOT NULL,
    content       TEXT NOT NULL,
    importance    REAL NOT NULL,
    metadata      TEXT NOT NULL,
    embedding     TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_session_ts ON events(session_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

CREATE TABLE IF NOT EXISTS facts (
    id               TEXT PRIMARY KEY,
    statement        TEXT NOT NULL,
    subject          TEXT,
    predicate        TEXT,
    object           TEXT,
    confidence       REAL NOT NULL,
    source_event_ids TEXT NOT NULL,
    created_ts       REAL NOT NULL,
    updated_ts       REAL NOT NULL,
    valid            INTEGER NOT NULL,
    superseded_by    TEXT,
    embedding        TEXT
);
CREATE INDEX IF NOT EXISTS idx_facts_valid ON facts(valid);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);

CREATE TABLE IF NOT EXISTS archive (
    id            TEXT PRIMARY KEY,
    summary       TEXT NOT NULL,
    origin_tier   TEXT NOT NULL,
    span_start    REAL NOT NULL,
    span_end      REAL NOT NULL,
    source_ids    TEXT NOT NULL,
    token_estimate INTEGER NOT NULL,
    embedding     TEXT
);
"""


class Database:
    """Thin, thread-safe wrapper around a SQLite connection."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        is_memory = self.path == ":memory:"
        if not is_memory:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage transactions explicitly
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._migrate()

    def _migrate(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            (current,) = self._conn.execute("PRAGMA user_version").fetchone()
            if current < SCHEMA_VERSION:
                self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # -- core ops ----------------------------------------------------------

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        with self._lock:
            self._conn.executemany(sql, rows)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return cast("sqlite3.Row | None", self._conn.execute(sql, params).fetchone())

    def count(self, table: str, where: str = "", params: Sequence[Any] = ()) -> int:
        clause = f" WHERE {where}" if where else ""
        row = self.query_one(f"SELECT COUNT(*) AS n FROM {table}{clause}", params)
        return int(row["n"]) if row else 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# (De)serialisation helpers shared by the stores
# ---------------------------------------------------------------------------


def dumps_embedding(vec: np.ndarray | list[float] | None) -> str | None:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32)
    return json.dumps([round(float(x), 7) for x in arr.tolist()])


def loads_embedding(blob: str | None) -> list[float] | None:
    if not blob:
        return None
    return list(json.loads(blob))


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def loads_json(blob: str | None, default: Any) -> Any:
    if not blob:
        return default
    return json.loads(blob)
