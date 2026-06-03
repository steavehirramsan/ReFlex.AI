"""Routing and prompt-assembly policy.

The :class:`Policy` concentrates the orchestrator's *decisions* — how important an input is,
when to reflect, and exactly how the prompt is assembled from persona, working memory,
retrieved context, and recent conversation. Isolating these here keeps the orchestrator a
thin control loop and makes the policy independently testable and tunable.
"""

from __future__ import annotations

import re

from ..config import ReflexConfig
from ..types import Event, Message, RetrievalBundle, Role

_IMPORTANT_CUE = re.compile(
    r"\b(remember|never forget|important|note that|my name is|i am|i'm|always|"
    r"deadline|goal|prefer|don't|do not)\b",
    re.IGNORECASE,
)


class Policy:
    """Encapsulates orchestration decisions and prompt construction."""

    def __init__(self, config: ReflexConfig) -> None:
        self.config = config

    # -- scoring -----------------------------------------------------------

    def importance_for_input(self, text: str) -> float:
        """Heuristic salience of a user input, used to prioritise it in memory."""
        score = 0.45
        if _IMPORTANT_CUE.search(text):
            score += 0.3
        if len(text.split()) > 30:
            score += 0.1
        if text.strip().endswith("?"):
            score -= 0.05  # questions are usually less durable than statements
        return max(0.0, min(1.0, score))

    def should_reflect(self, turn_index: int) -> bool:
        rc = self.config.reflection
        return rc.enabled and (turn_index % rc.every_n_turns == 0)

    # -- prompt assembly ---------------------------------------------------

    def build_messages(
        self,
        *,
        user_input: str,
        retrieval: RetrievalBundle,
        recent: list[Event],
        working_memory: Message | None,
    ) -> list[Message]:
        messages: list[Message] = [Message(role=Role.SYSTEM, content=self._system_prompt())]

        if working_memory is not None:
            messages.append(working_memory)

        if not retrieval.is_empty:
            messages.append(
                Message(
                    role=Role.SYSTEM,
                    content=(
                        "Relevant memory (each line: [tier:id score] content). Ground your "
                        "answer in these; do not invent facts beyond them:\n" + retrieval.render()
                    ),
                )
            )

        # Recent conversation, oldest-first, excluding internal reflection events.
        for ev in recent:
            if ev.kind == "reflection":
                continue
            role = ev.role if ev.role in (Role.USER, Role.ASSISTANT) else Role.USER
            messages.append(Message(role=role, content=ev.content))

        messages.append(Message(role=Role.USER, content=user_input))
        return messages

    def build_revision_messages(
        self,
        *,
        base_messages: list[Message],
        draft: str,
        flags: list[str],
    ) -> list[Message]:
        """Append a correction instruction so the model can repair a flagged draft."""
        critique = "\n".join(f"- {f}" for f in flags)
        instruction = (
            "Your previous draft was flagged by the integrity check for the issues below. "
            "Rewrite it to be fully grounded in the provided memory, remove any unsupported "
            "or contradictory claims, and state uncertainty plainly where memory is silent.\n\n"
            f"Draft:\n{draft}\n\nIssues:\n{critique}"
        )
        return [*base_messages, Message(role=Role.SYSTEM, content=instruction)]

    def _system_prompt(self) -> str:
        agent = self.config.agent
        lines = [agent.persona]
        if agent.goals:
            lines.append("Standing goals:\n" + "\n".join(f"- {g}" for g in agent.goals))
        return "\n\n".join(lines)
