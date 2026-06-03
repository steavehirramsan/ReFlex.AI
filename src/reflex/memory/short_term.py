"""Volatile tiers: the short-term buffer and the working-memory scratchpad.

These hold *what is happening now*. They live entirely in process memory and are bounded —
the short-term buffer by item count, working memory by a token budget tied to the model's
context window. When the buffer overflows, the memory manager is responsible for promoting
evicted items into the durable episodic store first.
"""

from __future__ import annotations

from collections import deque

from ..types import Event, Message


def estimate_tokens(text: str) -> int:
    """Cheap, model-agnostic token estimate (~4 chars/token). Good enough for budgeting."""
    return max(1, (len(text) + 3) // 4)


class ShortTermBuffer:
    """A bounded FIFO of the most recent events (raw recent input/output)."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: deque[Event] = deque(maxlen=capacity)

    def add(self, event: Event) -> Event | None:
        """Append an event; return the event evicted by capacity overflow, if any."""
        evicted: Event | None = None
        if len(self._items) == self.capacity:
            evicted = self._items[0]
        self._items.append(event)
        return evicted

    def recent(self, n: int | None = None) -> list[Event]:
        items = list(self._items)
        return items if n is None else items[-n:]

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


class WorkingMemory:
    """The active reasoning scratchpad bound to the context window.

    Holds the current goals plus a token-budgeted set of salient notes. When the budget is
    exceeded, the oldest notes are dropped (they remain durable in episodic memory).
    """

    def __init__(self, token_budget: int) -> None:
        self.token_budget = token_budget
        self.goals: list[str] = []
        self._notes: deque[tuple[str, int]] = deque()  # (text, token_cost)
        self._tokens = 0

    def set_goals(self, goals: list[str]) -> None:
        self.goals = list(goals)

    def note(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        cost = estimate_tokens(text)
        self._notes.append((text, cost))
        self._tokens += cost
        self._trim()

    def _trim(self) -> None:
        while self._tokens > self.token_budget and len(self._notes) > 1:
            _, cost = self._notes.popleft()
            self._tokens -= cost

    @property
    def tokens(self) -> int:
        return self._tokens

    def notes(self) -> list[str]:
        return [text for text, _ in self._notes]

    def render(self) -> str:
        """Render goals + notes as a system-message fragment."""
        parts: list[str] = []
        if self.goals:
            parts.append("Current goals:\n" + "\n".join(f"- {g}" for g in self.goals))
        if self._notes:
            parts.append("Working notes:\n" + "\n".join(f"- {t}" for t, _ in self._notes))
        return "\n\n".join(parts)

    def as_message(self) -> Message | None:
        rendered = self.render()
        from ..types import Role

        return Message(role=Role.SYSTEM, content=rendered) if rendered else None

    def clear(self) -> None:
        self._notes.clear()
        self._tokens = 0
