from __future__ import annotations

import pytest

from reflex.types import (
    Event,
    Flag,
    FlagKind,
    IntegrityReport,
    MemoryHit,
    MemoryTier,
    Message,
    RetrievalBundle,
    Role,
    new_id,
)


def test_new_id_prefix_and_uniqueness() -> None:
    a, b = new_id("evt"), new_id("evt")
    assert a.startswith("evt_") and b.startswith("evt_") and a != b


def test_message_as_dict() -> None:
    m = Message(role=Role.USER, content="hi", name="bob")
    assert m.as_dict() == {"role": "user", "content": "hi", "name": "bob"}


def test_event_defaults() -> None:
    e = Event(session_id="s", content="x")
    assert e.id.startswith("evt_")
    assert 0.0 <= e.importance <= 1.0
    assert e.role == Role.USER


def test_event_importance_bounds() -> None:
    with pytest.raises(ValueError):
        Event(session_id="s", content="x", importance=2.0)


def test_retrieval_bundle_render_and_empty() -> None:
    assert RetrievalBundle(query="q").is_empty
    bundle = RetrievalBundle(
        query="q",
        hits=[MemoryHit(record_id="r1", tier=MemoryTier.SEMANTIC, content="fact one", score=0.5)],
    )
    rendered = bundle.render()
    assert "semantic:r1" in rendered and "fact one" in rendered
    assert not bundle.is_empty


def test_retrieval_render_respects_budget() -> None:
    bundle = RetrievalBundle(
        query="q",
        hits=[
            MemoryHit(record_id=f"r{i}", tier=MemoryTier.EPISODIC, content="x" * 100, score=0.5)
            for i in range(10)
        ],
    )
    assert len(bundle.render(max_chars=150)) <= 200


def test_integrity_report_helpers() -> None:
    clean = IntegrityReport()
    assert clean.ok and clean.max_severity == 0.0

    flagged = IntegrityReport(
        score=0.3,
        flags=[Flag(kind=FlagKind.LOW_SUPPORT, severity=0.6, message="m")],
    )
    assert not flagged.ok
    assert flagged.max_severity == 0.6
    assert flagged.blocking(0.5) and not flagged.blocking(0.7)
