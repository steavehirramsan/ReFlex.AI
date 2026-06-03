"""Deterministic, offline LLM backend.

The mock backend lets the *entire* cognitive runtime — memory, retrieval, reflection,
integrity, the cognition loop — run and be tested with no GPU, no network, and no API key.
Its output is a deterministic function of the input messages, so tests are reproducible.

Two modes:

* **echo/grounded** (default): synthesises a reply that quotes the most relevant retrieved
  context, so the integrity layer's grounding checks behave realistically.
* **scripted**: pops successive canned replies from a queue, for tests that need to assert
  on an exact model output.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable

from ..types import Message, Role
from .base import LLMClient

_CONTEXT_LINE = re.compile(r"^\[[a-z_]+:[^\]]+\]\s*(.*)$")


class MockLLM(LLMClient):
    """A deterministic stand-in for a real chat model."""

    model = "reflex-mock"

    def __init__(self, scripted: Iterable[str] | None = None) -> None:
        self._scripted: deque[str] = deque(scripted or [])

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        if self._scripted:
            return self._scripted.popleft()
        return self._grounded_reply(messages)

    # -- internals ---------------------------------------------------------

    def _grounded_reply(self, messages: list[Message]) -> str:
        """Build a concise reply grounded in any retrieved context block.

        Quoting the retrieved memory keeps the integrity layer's support score high for
        well-grounded turns, which is exactly the behaviour a real, well-behaved model has.
        Snippets are deduplicated and truncated so multi-turn output stays readable.
        """
        user_msg = next((m.content for m in reversed(messages) if m.role == Role.USER), "")
        context_snippets = self._extract_context(messages)

        if context_snippets:
            grounding = "; ".join(context_snippets[:2])
            return f"Based on memory: {grounding}. (In response to “{user_msg.strip()}”.)"
        return (
            f"I don't have anything in memory about that yet. Regarding "
            f"“{user_msg.strip()}”, I'll note it and follow up."
        ).strip()

    @staticmethod
    def _extract_context(messages: list[Message], *, max_len: int = 160) -> list[str]:
        """Pull deduplicated, truncated memory lines out of system/context messages.

        Prefers genuine memory snippets over the mock's own prior echoes, so replies don't
        compound into noise across turns.
        """
        seen: set[str] = set()
        snippets: list[str] = []
        for msg in messages:
            if msg.role not in (Role.SYSTEM, Role.OBSERVATION):
                continue
            for line in msg.content.splitlines():
                m = _CONTEXT_LINE.match(line.strip())
                if not (m and m.group(1)):
                    continue
                snippet = m.group(1).strip()
                if snippet.startswith(("Based on memory:", "I don't have anything in memory")):
                    continue  # skip the mock's own earlier output
                snippet = snippet[:max_len].rstrip()
                if snippet and snippet not in seen:
                    seen.add(snippet)
                    snippets.append(snippet)
        return snippets
