from __future__ import annotations

from reflex.config import ReflexConfig
from reflex.eval import RetentionBenchmark, RetentionResult, run_retention
from reflex.runtime.agent import Agent


def _cfg() -> ReflexConfig:
    return ReflexConfig.load(
        env=False,
        overrides={
            "memory": {"db_path": ":memory:", "vector": {"backend": "numpy"}},
            "embeddings": {"dim": 256},
            "logging": {"rich": False, "level": "ERROR"},
        },
    )


async def test_retention_recalls_planted_facts() -> None:
    bench = RetentionBenchmark(distractors=30, top_k=10, seed=7)
    async with Agent.from_config(_cfg()) as agent:
        result = await bench.run(agent)
    assert isinstance(result, RetentionResult)
    assert result.total == len(bench.items)
    # Salient facts are promoted to the durable semantic tier, so retention is strong.
    assert result.retention_rate >= 0.9
    assert "retention" in result.render().lower()


async def test_retention_survives_compaction() -> None:
    # Enough distractors to trigger episodic compaction; semantic facts must still survive.
    cfg = _cfg()
    cfg.memory.compaction.episodic_threshold = 40
    cfg.memory.compaction.batch_size = 20
    bench = RetentionBenchmark(distractors=80, top_k=10, seed=3)
    async with Agent.from_config(cfg) as agent:
        result = await bench.run(agent)
        assert agent.memory.archive.count() > 0  # compaction actually happened
    assert result.retention_rate >= 0.9


async def test_retention_is_deterministic() -> None:
    r1 = await run_retention(_cfg(), distractors=20, top_k=8, seed=99)
    r2 = await run_retention(_cfg(), distractors=20, top_k=8, seed=99)
    assert r1.per_item == r2.per_item


def test_result_metrics() -> None:
    res = RetentionResult(
        total=4, recalled=3, distractors=10, top_k=5, per_item=[True, True, True, False]
    )
    assert res.retention_rate == 0.75
