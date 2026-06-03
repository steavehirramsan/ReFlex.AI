from __future__ import annotations

from reflex.config import IntegrityConfig
from reflex.embeddings.hashing import HashingEmbedder
from reflex.integrity.guard import IntegrityGuard, _polarity
from reflex.types import MemoryHit, MemoryTier, RetrievalBundle


def _bundle(*contents: str) -> RetrievalBundle:
    return RetrievalBundle(
        query="q",
        hits=[
            MemoryHit(record_id=f"e{i}", tier=MemoryTier.EPISODIC, content=c, score=0.9)
            for i, c in enumerate(contents)
        ],
    )


def _guard() -> IntegrityGuard:
    return IntegrityGuard(IntegrityConfig(), HashingEmbedder(dim=256))


def test_polarity_detection() -> None:
    assert _polarity("the server is running") is True
    assert _polarity("the server is not running") is False


def test_grounded_claim_is_clean(memory) -> None:
    guard = _guard()
    report = guard.check(
        "The sky is blue and clear today.",
        _bundle("the sky is blue and clear"),
        memory,
    )
    assert report.ok
    assert report.score == 1.0


def test_fabricated_memory_flag(memory) -> None:
    guard = _guard()
    report = guard.check(
        "You told me earlier that the meeting is scheduled at noon.",
        RetrievalBundle(query="q"),  # nothing retrieved
        memory,
    )
    kinds = {f.kind.value for f in report.flags}
    assert "fabricated_memory" in kinds
    assert report.score < 1.0


def test_low_support_flag(memory) -> None:
    guard = _guard()
    report = guard.check(
        "The database shards writes across three availability zones.",
        _bundle("my favorite dessert is tiramisu"),  # present but irrelevant context
        memory,
    )
    assert any(f.kind.value == "low_support" for f in report.flags)


def test_factual_drift_against_established_fact(memory) -> None:
    memory.add_fact("The production server is running", confidence=0.95)
    guard = _guard()
    report = guard.check(
        "The production server is not running at all.",
        _bundle("the production server is running"),
        memory,
    )
    assert any(f.kind.value == "factual_drift" for f in report.flags)


def test_self_contradiction_flag(memory) -> None:
    guard = _guard()
    report = guard.check(
        "The deployment is fully secure. The deployment is not secure.",
        RetrievalBundle(query="q"),
        memory,
    )
    assert any(f.kind.value == "inconsistent_output" for f in report.flags)


def test_disabled_guard_returns_clean(memory) -> None:
    guard = IntegrityGuard(IntegrityConfig(enabled=False), HashingEmbedder(dim=64))
    report = guard.check(
        "You said earlier nonsense never grounded.", RetrievalBundle(query="q"), memory
    )
    assert report.ok


def test_questions_and_hedges_not_flagged(memory) -> None:
    guard = _guard()
    report = guard.check(
        "What did you say earlier about the budget? I'm not sure I remember the figure.",
        RetrievalBundle(query="q"),
        memory,
    )
    assert report.ok  # a question + a hedge are not asserted claims


def test_blocking_threshold() -> None:
    from reflex.types import Flag, FlagKind, IntegrityReport

    report = IntegrityReport(
        score=0.1, flags=[Flag(kind=FlagKind.FABRICATED_MEMORY, severity=0.85, message="m")]
    )
    assert report.blocking(0.8) is True
    assert report.blocking(0.9) is False
