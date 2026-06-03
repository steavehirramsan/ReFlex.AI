"""The long-running :class:`Agent` — the public, batteries-included entry point.

``Agent`` assembles every subsystem from a :class:`~reflex.config.ReflexConfig` and exposes a
small, stable surface: :meth:`turn` (full auditable record), :meth:`chat` (just the reply),
plus explicit memory helpers (:meth:`remember`, :meth:`recall`). It is designed to be held
open across a long deployment — its memory persists to disk, so a restart resumes the same
durable state.

Use it as an async context manager so backends (HTTP clients, the SQLite connection) are
released deterministically::

    async with Agent.from_config(cfg) as agent:
        print(await agent.chat("hello"))
"""

from __future__ import annotations

from types import TracebackType

from ..config import ReflexConfig
from ..core.orchestrator import Orchestrator
from ..core.policy import Policy
from ..embeddings import Embedder, build_embedder
from ..integrity.guard import IntegrityGuard
from ..llm import LLMClient, build_llm
from ..logging import configure_logging, get_logger
from ..memory.manager import MemoryManager
from ..reflection.engine import ReflectionEngine
from ..types import AgentTurn, Fact, RetrievalBundle

log = get_logger(__name__)


class Agent:
    """A persistent, self-correcting agent with tiered memory."""

    def __init__(
        self,
        config: ReflexConfig,
        *,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        memory: MemoryManager | None = None,
    ) -> None:
        configure_logging(config.logging.level, use_rich=config.logging.rich)
        self.config = config
        self.embedder = embedder or build_embedder(config.embeddings)
        self.llm = llm or build_llm(config.llm)
        self.memory = memory or MemoryManager(config.memory, self.embedder)
        self.memory.working.set_goals(config.agent.goals)

        self.reflection = ReflectionEngine(config.reflection, self.llm)
        self.integrity = IntegrityGuard(config.integrity, self.embedder)
        self.orchestrator = Orchestrator(
            config,
            llm=self.llm,
            memory=self.memory,
            reflection=self.reflection,
            integrity=self.integrity,
            policy=Policy(config),
        )
        self.session_id = config.agent.session_id or self.memory.new_session_id()
        log.info(
            "Agent '%s' ready (llm=%s, embedder dim=%d, session=%s)",
            config.agent.name,
            self.llm.model,
            self.embedder.dim,
            self.session_id,
        )

    @classmethod
    def from_config(cls, config: ReflexConfig | None = None) -> Agent:
        """Build an agent from a config (or all defaults: offline mock + local SQLite)."""
        return cls(config or ReflexConfig())

    # -- interaction -------------------------------------------------------

    async def turn(self, user_input: str, *, session_id: str | None = None) -> AgentTurn:
        """Process an input and return the full, auditable :class:`AgentTurn`."""
        return await self.orchestrator.handle(user_input, session_id=session_id or self.session_id)

    async def chat(self, user_input: str, *, session_id: str | None = None) -> str:
        """Convenience wrapper around :meth:`turn` that returns just the response text."""
        return (await self.turn(user_input, session_id=session_id)).response

    # -- explicit memory ---------------------------------------------------

    def remember(self, statement: str, *, confidence: float = 0.9) -> Fact:
        """Write a durable fact directly into semantic memory."""
        return self.memory.add_fact(statement, confidence=confidence)

    def recall(self, query: str, *, total_k: int | None = None) -> RetrievalBundle:
        """Retrieve relevant memory for a query without generating a response."""
        return self.memory.retrieve(query, total_k=total_k)

    def stats(self) -> dict[str, int]:
        return self.memory.stats()

    # -- lifecycle ---------------------------------------------------------

    async def aclose(self) -> None:
        await self.llm.aclose()
        self.memory.close()

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
