"""The orchestrator — the cognitive control loop.

Every input is routed through the same pipeline before any response is emitted:

1. **retrieve** relevant memory for the input;
2. **assemble** the prompt from persona, working memory, retrieved context, and recent
   conversation;
3. **generate** a candidate response from the LLM;
4. **verify** it with the integrity layer, optionally **revising** a flagged draft;
5. **persist** the exchange and **reflect** to write durable corrections/facts;
6. **compact** memory if a tier has overflowed.

The orchestrator is async because LLM calls are the only real I/O; memory operations are
fast, synchronous, local calls.
"""

from __future__ import annotations

import time

from ..config import ReflexConfig
from ..errors import IntegrityViolation
from ..integrity.guard import IntegrityGuard
from ..llm.base import LLMClient
from ..logging import get_logger
from ..memory.manager import MemoryManager
from ..reflection.engine import ReflectionEngine
from ..types import AgentTurn, IntegrityReport, Message, RetrievalBundle, Role
from .policy import Policy

log = get_logger(__name__)

_PROMPT_RECENT_TURNS = 8


class Orchestrator:
    """Routes inputs through memory, generation, integrity, and reflection."""

    def __init__(
        self,
        config: ReflexConfig,
        llm: LLMClient,
        memory: MemoryManager,
        reflection: ReflectionEngine,
        integrity: IntegrityGuard,
        policy: Policy | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.memory = memory
        self.reflection = reflection
        self.integrity = integrity
        self.policy = policy or Policy(config)
        self._turn_index = 0

    async def handle(self, user_input: str, *, session_id: str) -> AgentTurn:
        """Process one input end-to-end and return the auditable turn record."""
        started = time.perf_counter()
        self._turn_index += 1

        # 1–2. Retrieve and assemble (before recording, so the input can't retrieve itself).
        retrieval = self.memory.retrieve(user_input)
        recent = self.memory.episodic.recent(session_id, _PROMPT_RECENT_TURNS)
        user_event = self.memory.record_event(
            session_id,
            user_input,
            role=Role.USER,
            importance=self.policy.importance_for_input(user_input),
        )
        base_messages = self.policy.build_messages(
            user_input=user_input,
            retrieval=retrieval,
            recent=recent,
            working_memory=self.memory.working.as_message(),
        )

        # 3–4. Generate, verify, and (optionally) revise.
        draft = await self.llm.complete(base_messages)
        report = self.integrity.check(draft, retrieval, self.memory)
        draft, report, revised = await self._enforce_integrity(
            base_messages, draft, report, retrieval
        )

        # 5. Persist the response.
        assistant_event = self.memory.record_event(
            session_id,
            draft,
            role=Role.ASSISTANT,
            importance=0.5 + 0.2 * (1.0 - report.score),
            metadata={"integrity_score": report.score, "revised": revised},
        )

        # 5b. Reflect (self-correction) and write durable state.
        reflection_result = None
        if self.policy.should_reflect(self._turn_index):
            reflection_result = await self.reflection.reflect(
                session_id=session_id,
                user_input=user_input,
                response=draft,
                integrity=report,
                memory=self.memory,
                source_event_ids=[user_event.id, assistant_event.id],
            )

        # 6. Keep durable memory bounded.
        self.memory.maybe_compact()

        turn = AgentTurn(
            session_id=session_id,
            user_input=user_input,
            response=draft,
            retrieval=retrieval,
            integrity=report,
            reflection=reflection_result,
            revised=revised,
            latency_s=round(time.perf_counter() - started, 4),
        )
        log.debug(
            "turn %d: integrity=%.2f revised=%s hits=%d latency=%.3fs",
            self._turn_index,
            report.score,
            revised,
            len(retrieval.hits),
            turn.latency_s,
        )
        return turn

    async def _enforce_integrity(
        self,
        base_messages: list[Message],
        draft: str,
        report: IntegrityReport,
        retrieval: RetrievalBundle,
    ) -> tuple[str, IntegrityReport, bool]:
        """Apply the configured integrity policy to a flagged draft."""
        ic = self.config.integrity
        if report.ok or not report.blocking(ic.block_threshold):
            return draft, report, False

        if ic.on_violation == "raise":
            raise IntegrityViolation(
                "Response failed integrity check",
                flags=[f.message for f in report.flags],
            )
        if ic.on_violation == "flag":
            return draft, report, False

        # on_violation == "revise": iteratively repair the draft.
        revised = False
        for attempt in range(ic.max_revisions):
            flags = [f"[{f.kind.value}] {f.message}" for f in report.flags]
            rev_messages = self.policy.build_revision_messages(
                base_messages=base_messages, draft=draft, flags=flags
            )
            new_draft = await self.llm.complete(rev_messages)
            new_report = self.integrity.check(new_draft, retrieval, self.memory)
            revised = True
            draft, report = new_draft, new_report
            log.debug("integrity revision %d -> score %.2f", attempt + 1, report.score)
            if report.ok or not report.blocking(ic.block_threshold):
                break
        return draft, report, revised

    @property
    def turn_index(self) -> int:
        return self._turn_index
