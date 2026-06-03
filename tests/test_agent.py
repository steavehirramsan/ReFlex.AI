from __future__ import annotations

import pytest

from reflex.config import ReflexConfig
from reflex.errors import IntegrityViolation
from reflex.llm.mock import MockLLM
from reflex.runtime.agent import Agent


def _cfg(**over) -> ReflexConfig:
    base = {
        "memory": {"db_path": ":memory:", "vector": {"backend": "numpy"}},
        "embeddings": {"dim": 128},
        "logging": {"rich": False, "level": "WARNING"},
    }
    base.update(over)
    return ReflexConfig.load(env=False, overrides=base)


async def test_turn_records_user_and_assistant(config: ReflexConfig) -> None:
    async with Agent.from_config(config) as agent:
        turn = await agent.turn("hello there friend")
        assert turn.response
        assert agent.memory.episodic.count() >= 2  # user + assistant (+ reflection)


async def test_memory_carries_across_turns(config: ReflexConfig) -> None:
    async with Agent.from_config(config) as agent:
        await agent.turn("Remember that my project is called ReFlex.")
        turn = await agent.turn("What is my project called?")
        # The semantic fact should be retrieved on the follow-up turn.
        assert any("ReFlex" in h.content for h in turn.retrieval.hits)


async def test_remember_and_recall(config: ReflexConfig) -> None:
    async with Agent.from_config(config) as agent:
        agent.remember("The mascot is a tortoise that never forgets")
        bundle = agent.recall("what is the mascot")
        assert any("tortoise" in h.content for h in bundle.hits)


async def test_integrity_revise_path() -> None:
    cfg = _cfg(
        reflection={"enabled": False}, integrity={"on_violation": "revise", "max_revisions": 1}
    )
    llm = MockLLM(
        scripted=[
            "You told me earlier that the secret code is 42.",  # fabricated memory -> blocked
            "I don't have any record of a secret code in memory.",  # clean revision
        ]
    )
    async with Agent(cfg, llm=llm) as agent:
        turn = await agent.turn("what was the secret code?")
        assert turn.revised is True
        assert "don't have any record" in turn.response


async def test_integrity_raise_path() -> None:
    cfg = _cfg(reflection={"enabled": False}, integrity={"on_violation": "raise"})
    llm = MockLLM(scripted=["You previously said the launch is tomorrow at dawn."])
    async with Agent(cfg, llm=llm) as agent:
        with pytest.raises(IntegrityViolation):
            await agent.turn("when is the launch?")


async def test_integrity_flag_path_keeps_response() -> None:
    cfg = _cfg(reflection={"enabled": False}, integrity={"on_violation": "flag"})
    llm = MockLLM(scripted=["You told me earlier the meeting moved to Friday."])
    async with Agent(cfg, llm=llm) as agent:
        turn = await agent.turn("when is the meeting?")
        assert turn.revised is False
        assert turn.integrity.flags  # flagged but not blocked


async def test_chat_returns_text(config: ReflexConfig) -> None:
    async with Agent.from_config(config) as agent:
        out = await agent.chat("say something")
        assert isinstance(out, str) and out


async def test_turn_index_increments(config: ReflexConfig) -> None:
    async with Agent.from_config(config) as agent:
        await agent.turn("one")
        await agent.turn("two")
        assert agent.orchestrator.turn_index == 2


async def test_session_persistence_across_restart(tmp_path) -> None:
    db = str(tmp_path / "agent.db")
    cfg = _cfg(memory={"db_path": db, "vector": {"backend": "numpy"}})
    async with Agent(cfg) as agent:
        agent.remember("The archived design doc lives in the wiki under /reflex/arch")
    # Re-open a fresh agent on the same DB; durable memory should survive.
    async with Agent(cfg) as agent2:
        bundle = agent2.recall("where is the design doc")
        assert any("wiki" in h.content for h in bundle.hits)
