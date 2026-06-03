"""Core domain types for ReFlex.

These models are the lingua franca between subsystems (memory, reflection, integrity,
orchestration). They are deliberately backend-agnostic: an :class:`Event` looks the same
whether it lives in SQLite, Postgres, or an in-memory store.

Design notes
------------
* Timestamps are epoch seconds (``float``) produced by :func:`now`. A single clock
  function keeps tests deterministic (it can be monkeypatched) and avoids timezone drift.
* Embeddings are stored as plain ``list[float]`` so every record is trivially JSON- and
  DB-serializable. Conversion to ``numpy`` happens only at the vector-index boundary.
* IDs are prefixed UUIDs (``evt_…``, ``fact_…``) so a bare ID is self-describing in logs.
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

Embedding = list[float]


def now() -> float:
    """Current wall-clock time in epoch seconds.

    Centralised so tests can monkeypatch a single symbol to freeze time.
    """
    return time.time()


def new_id(prefix: str) -> str:
    """Generate a short, self-describing identifier, e.g. ``evt_1f3c…``."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class Role(StrEnum):
    """Speaker role for a conversational message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    OBSERVATION = "observation"


class MemoryTier(StrEnum):
    """The explicit tiers of the memory hierarchy (see README §Memory Subsystem)."""

    SHORT_TERM = "short_term"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    ARCHIVE = "archive"


class FlagKind(StrEnum):
    """Categories of issue the integrity layer can raise."""

    FACTUAL_DRIFT = "factual_drift"
    FABRICATED_MEMORY = "fabricated_memory"
    INVALID_REASONING = "invalid_reasoning"
    INCONSISTENT_OUTPUT = "inconsistent_output"
    LOW_SUPPORT = "low_support"


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single conversational message exchanged with the LLM."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str
    name: str | None = None

    def as_dict(self) -> dict[str, str]:
        """Render to the OpenAI chat-message shape."""
        d = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


# ---------------------------------------------------------------------------
# Memory records
# ---------------------------------------------------------------------------


class Event(BaseModel):
    """An episodic record: *what happened, and when*.

    Events are append-only. They are never mutated after creation (only demoted to the
    archive), which keeps the episodic log a faithful history.
    """

    id: str = Field(default_factory=lambda: new_id("evt"))
    session_id: str
    ts: float = Field(default_factory=now)
    kind: str = "message"  # message | action | observation | reflection | correction
    role: Role = Role.USER
    content: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Embedding | None = None


class Fact(BaseModel):
    """A semantic record: *what is true*.

    Facts are distilled, deduplicated assertions. Unlike events they are mutable: a fact
    can be invalidated (``valid=False``) and superseded by a newer fact when the integrity
    layer detects that reality changed.
    """

    id: str = Field(default_factory=lambda: new_id("fact"))
    statement: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    source_event_ids: list[str] = Field(default_factory=list)
    created_ts: float = Field(default_factory=now)
    updated_ts: float = Field(default_factory=now)
    valid: bool = True
    superseded_by: str | None = None
    embedding: Embedding | None = None


class ArchiveRecord(BaseModel):
    """A compressed, cold-tier summary spanning many events or facts."""

    id: str = Field(default_factory=lambda: new_id("arc"))
    summary: str
    origin_tier: MemoryTier = MemoryTier.EPISODIC
    span_start: float = Field(default_factory=now)
    span_end: float = Field(default_factory=now)
    source_ids: list[str] = Field(default_factory=list)
    token_estimate: int = 0
    embedding: Embedding | None = None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class MemoryHit(BaseModel):
    """A single scored retrieval result with provenance back to its tier."""

    record_id: str
    tier: MemoryTier
    content: str
    score: float
    ts: float = Field(default_factory=now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalBundle(BaseModel):
    """The merged, ranked context retrieved for a single query."""

    query: str
    hits: list[MemoryHit] = Field(default_factory=list)

    def render(self, max_chars: int = 4000) -> str:
        """Render hits into a compact, citable context block for the prompt."""
        lines: list[str] = []
        budget = max_chars
        for hit in self.hits:
            line = f"[{hit.tier.value}:{hit.record_id} score={hit.score:.2f}] {hit.content}"
            if len(line) > budget:
                break
            lines.append(line)
            budget -= len(line) + 1
        return "\n".join(lines)

    @property
    def is_empty(self) -> bool:
        return not self.hits


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


class Flag(BaseModel):
    """A single integrity concern raised about a candidate response."""

    kind: FlagKind
    severity: float = Field(ge=0.0, le=1.0)
    message: str
    evidence: list[str] = Field(default_factory=list)


class IntegrityReport(BaseModel):
    """The verdict of the integrity layer for one candidate response."""

    score: float = Field(default=1.0, ge=0.0, le=1.0)  # 1.0 == clean
    flags: list[Flag] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.flags

    def blocking(self, threshold: float) -> bool:
        """True if any flag is severe enough to block the response."""
        return any(f.severity >= threshold for f in self.flags)

    @property
    def max_severity(self) -> float:
        return max((f.severity for f in self.flags), default=0.0)


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------


class ReflectionResult(BaseModel):
    """The output of one pass of the reflection (self-correction) engine."""

    summary: str = ""
    drift_detected: bool = False
    corrections: list[str] = Field(default_factory=list)
    new_facts: list[str] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Turn-level result
# ---------------------------------------------------------------------------


class AgentTurn(BaseModel):
    """The complete, auditable record of one input → response cycle."""

    id: str = Field(default_factory=lambda: new_id("turn"))
    session_id: str
    ts: float = Field(default_factory=now)
    user_input: str
    response: str
    retrieval: RetrievalBundle
    integrity: IntegrityReport
    reflection: ReflectionResult | None = None
    revised: bool = False
    latency_s: float = 0.0
