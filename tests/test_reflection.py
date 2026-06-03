from __future__ import annotations

from reflex.config import ReflectionConfig
from reflex.llm.mock import MockLLM
from reflex.reflection.engine import ReflectionEngine, _is_salient
from reflex.types import Flag, FlagKind, IntegrityReport


def test_is_salient_distinguishes_facts_from_chitchat() -> None:
    assert _is_salient("The staging cluster lives in region eu-west-2")  # has a code
    assert _is_salient("My manager's name is Priya Raman")  # proper noun
    assert not _is_salient("the afternoon went smoothly overall")  # vague
    assert not _is_salient("what is the budget?")  # question
    assert not _is_salient("ok thanks")  # too short


async def _reflect(engine: ReflectionEngine, memory, user_input: str, integrity=None):
    return await engine.reflect(
        session_id="s",
        user_input=user_input,
        response="ok",
        integrity=integrity or IntegrityReport(),
        memory=memory,
    )


async def test_heuristic_extracts_remember_clause(memory) -> None:
    engine = ReflectionEngine(ReflectionConfig())
    result = await _reflect(engine, memory, "Please remember that the launch date is in March.")
    assert any("launch date is in March" in f for f in result.new_facts)
    assert memory.semantic.count() >= 1


async def test_heuristic_extracts_my_attr(memory) -> None:
    engine = ReflectionEngine(ReflectionConfig())
    result = await _reflect(engine, memory, "My favorite database is PostgreSQL.")
    assert any("favorite database is PostgreSQL" in f for f in result.new_facts)


async def test_disabled_reflection_is_noop(memory) -> None:
    engine = ReflectionEngine(ReflectionConfig(enabled=False))
    result = await _reflect(engine, memory, "Remember that x is y.")
    assert result.new_facts == []
    assert memory.semantic.count() == 0


async def test_integrity_flags_become_corrections(memory) -> None:
    engine = ReflectionEngine(ReflectionConfig())
    report = IntegrityReport(
        score=0.2,
        flags=[Flag(kind=FlagKind.FACTUAL_DRIFT, severity=0.9, message="contradiction")],
    )
    result = await _reflect(engine, memory, "nothing notable here", integrity=report)
    assert result.drift_detected is True
    assert any("factual_drift" in c for c in result.corrections)


async def test_reflection_writes_reflection_event(memory) -> None:
    engine = ReflectionEngine(ReflectionConfig())
    before = memory.episodic.count()
    await _reflect(engine, memory, "Remember that the API key rotates weekly.")
    after = memory.episodic.count()
    assert after == before + 1  # a reflection event was recorded


async def test_llm_fact_augmentation_parses_fact_lines(memory) -> None:
    llm = MockLLM(scripted=["FACT: The server lives in us-east-1.\nFACT: TLS is required."])
    engine = ReflectionEngine(ReflectionConfig(), llm=llm)
    result = await _reflect(engine, memory, "some chatter without explicit facts")
    assert any("us-east-1" in f for f in result.new_facts)
    assert any("TLS is required" in f for f in result.new_facts)


async def test_llm_failure_falls_back_to_heuristics(memory) -> None:
    class BoomLLM(MockLLM):
        async def complete(self, *a, **k):  # type: ignore[override]
            raise RuntimeError("backend down")

    engine = ReflectionEngine(ReflectionConfig(), llm=BoomLLM())
    result = await _reflect(engine, memory, "Remember that backups run nightly.")
    assert any("backups run nightly" in f for f in result.new_facts)
