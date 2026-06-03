"""ReFlex.AI — persistent cognitive architecture for long-running AI agents.

Public API
----------
>>> from reflex import ReflexConfig, Agent
>>> import asyncio
>>> agent = Agent.from_config(ReflexConfig.load())  # defaults: offline mock + SQLite memory
>>> asyncio.run(agent.chat("Remember that my project is called ReFlex."))

The high-level :class:`~reflex.runtime.agent.Agent` wires together the orchestrator, the
tiered :class:`~reflex.memory.MemoryManager`, the reflection engine, and the integrity layer.
Every subsystem is also importable and usable on its own.
"""

from __future__ import annotations

from .config import ReflexConfig
from .errors import (
    BackendError,
    ConfigError,
    IntegrityViolation,
    LLMError,
    ReflexError,
    ReflexMemoryError,
)
from .runtime.agent import Agent
from .types import (
    AgentTurn,
    ArchiveRecord,
    Event,
    Fact,
    Flag,
    IntegrityReport,
    MemoryHit,
    MemoryTier,
    Message,
    ReflectionResult,
    RetrievalBundle,
    Role,
)
from .version import __version__

__all__ = [
    "Agent",
    "AgentTurn",
    "ArchiveRecord",
    "BackendError",
    "ConfigError",
    "Event",
    "Fact",
    "Flag",
    "IntegrityReport",
    "IntegrityViolation",
    "LLMError",
    "MemoryHit",
    "MemoryTier",
    "Message",
    "ReflectionResult",
    "ReflexConfig",
    "ReflexError",
    "ReflexMemoryError",
    "RetrievalBundle",
    "Role",
    "__version__",
]
