"""Language-model backend interface.

Every LLM backend implements :class:`LLMClient`. The orchestrator depends only on this
interface, so swapping a deterministic offline mock for a ROCm vLLM server is a config
change, never a code change.
"""

from __future__ import annotations

import abc

from ..types import Message


class LLMClient(abc.ABC):
    """Abstract async chat-completion backend."""

    #: Human-readable model identifier, surfaced in logs and turn metadata.
    model: str = "unknown"

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        """Return the assistant completion for ``messages``.

        Implementations must be safe to call concurrently and should raise
        :class:`reflex.errors.LLMError` on unrecoverable backend failures.
        """
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release any resources (HTTP clients, model handles). Default: no-op."""
        return None
