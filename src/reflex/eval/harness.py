"""Evaluation harness — reproducible measurement of memory behaviour.

The README commits to *honest, reproducible numbers*. This module provides the first
benchmark in that suite: **memory retention under session growth**. It plants a set of
target facts, floods the agent with unrelated distractor turns to grow the durable stores
well past a single context window, and then probes whether each planted fact is still
retrievable.

Because it runs on the deterministic offline backends by default, the benchmark is fully
reproducible: same config + same seed ⇒ same numbers. Swap in a real model/embedder via
config to measure a production setup on identical inputs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..config import ReflexConfig
from ..logging import get_logger
from ..runtime.agent import Agent

log = get_logger(__name__)


@dataclass(frozen=True)
class RetentionItem:
    """One planted fact and the query used to probe its later recall."""

    fact: str
    probe: str
    answer: str  # substring that must appear in a retrieved hit to count as recalled


@dataclass
class RetentionResult:
    """Aggregate metrics from a retention run."""

    total: int
    recalled: int
    distractors: int
    top_k: int
    per_item: list[bool] = field(default_factory=list)

    @property
    def retention_rate(self) -> float:
        return self.recalled / self.total if self.total else 0.0

    def render(self) -> str:
        return (
            f"Memory retention @ k={self.top_k}: {self.recalled}/{self.total} "
            f"({self.retention_rate:.1%}) recalled after {self.distractors} distractor turns."
        )


# A small, deterministic synthetic dataset. Each fact is lexically distinct so retrieval
# quality — not luck — determines whether it is recalled.
_DEFAULT_ITEMS: list[RetentionItem] = [
    RetentionItem(
        "My passport number is X7741 stored for travel", "what is my passport number", "X7741"
    ),
    RetentionItem(
        "The staging cluster lives in region eu-west-2", "where is the staging cluster", "eu-west-2"
    ),
    RetentionItem("Our incident hotline is extension 5582", "what is the incident hotline", "5582"),
    RetentionItem(
        "The encryption key rotates every 90 days",
        "how often does the encryption key rotate",
        "90 days",
    ),
    RetentionItem("My manager's name is Priya Raman", "who is my manager", "Priya Raman"),
    RetentionItem(
        "The data warehouse uses the Iceberg table format",
        "what table format does the warehouse use",
        "Iceberg",
    ),
    RetentionItem(
        "Release trains depart every second Thursday",
        "when do release trains depart",
        "second Thursday",
    ),
    RetentionItem(
        "The mascot's name is Sheldon the tortoise", "what is the mascot's name", "Sheldon"
    ),
]

# Distractors are deliberately vague chit-chat: no numbers, no proper nouns. A good fact
# extractor should ignore them, so retention reflects real signal, not lexical luck.
_DISTRACTOR_TOPICS = [
    "the weather felt pleasant and calm",
    "lunch in the cafeteria was quite tasty",
    "the afternoon meeting went smoothly overall",
    "the office plants looked healthy again",
    "everything seemed calm and quiet today",
    "the morning passed without any trouble",
]


class RetentionBenchmark:
    """Measures whether planted facts survive heavy session growth."""

    def __init__(
        self,
        items: list[RetentionItem] | None = None,
        *,
        distractors: int = 100,
        top_k: int = 10,
        seed: int = 1234,
    ) -> None:
        self.items = items or _DEFAULT_ITEMS
        self.distractors = distractors
        self.top_k = top_k
        self._rng = random.Random(seed)

    async def run(self, agent: Agent) -> RetentionResult:
        # 1. Plant the target facts.
        for item in self.items:
            await agent.turn(item.fact)

        # 2. Grow the stores far past one context window with unrelated chatter.
        #    Phrased without numbers or proper nouns so the extractor rightly ignores them.
        for _ in range(self.distractors):
            topic = self._rng.choice(_DISTRACTOR_TOPICS)
            await agent.turn(f"By the way, {topic}.")

        # 3. Probe recall of each planted fact.
        per_item: list[bool] = []
        for item in self.items:
            bundle = agent.recall(item.probe, total_k=self.top_k)
            hit = any(item.answer.lower() in h.content.lower() for h in bundle.hits)
            per_item.append(hit)

        result = RetentionResult(
            total=len(self.items),
            recalled=sum(per_item),
            distractors=self.distractors,
            top_k=self.top_k,
            per_item=per_item,
        )
        log.info(result.render())
        return result


async def run_retention(
    config: ReflexConfig | None = None,
    *,
    distractors: int = 100,
    top_k: int = 10,
    seed: int = 1234,
) -> RetentionResult:
    """Convenience entry point: build an agent from config and run the retention benchmark."""
    cfg = config or ReflexConfig.load(overrides={"memory": {"db_path": ":memory:"}})
    benchmark = RetentionBenchmark(distractors=distractors, top_k=top_k, seed=seed)
    async with Agent.from_config(cfg) as agent:
        return await benchmark.run(agent)
