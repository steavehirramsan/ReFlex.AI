from __future__ import annotations

import pytest

from reflex.memory.short_term import ShortTermBuffer, WorkingMemory, estimate_tokens
from reflex.types import Event, Role


def _event(content: str) -> Event:
    return Event(session_id="s", content=content)


def test_estimate_tokens_monotonic() -> None:
    assert estimate_tokens("a") >= 1
    assert estimate_tokens("a" * 40) > estimate_tokens("a" * 4)


def test_buffer_evicts_oldest() -> None:
    buf = ShortTermBuffer(capacity=2)
    assert buf.add(_event("one")) is None
    assert buf.add(_event("two")) is None
    evicted = buf.add(_event("three"))
    assert evicted is not None and evicted.content == "one"
    assert len(buf) == 2
    assert [e.content for e in buf.recent()] == ["two", "three"]


def test_buffer_recent_n() -> None:
    buf = ShortTermBuffer(capacity=5)
    for c in "abcde":
        buf.add(_event(c))
    assert [e.content for e in buf.recent(2)] == ["d", "e"]


def test_buffer_capacity_validation() -> None:
    with pytest.raises(ValueError):
        ShortTermBuffer(capacity=0)


def test_working_memory_goals_and_notes() -> None:
    wm = WorkingMemory(token_budget=1000)
    wm.set_goals(["ship reflex", "stay grounded"])
    wm.note("user prefers concise answers")
    rendered = wm.render()
    assert "ship reflex" in rendered
    assert "concise" in rendered


def test_working_memory_token_budget_trims() -> None:
    wm = WorkingMemory(token_budget=5)  # ~20 chars
    wm.note("first note that is quite long and exceeds budget")
    wm.note("second note also reasonably long here")
    # Only the most recent note survives once the budget is blown.
    assert len(wm.notes()) == 1
    assert wm.tokens <= max(5, estimate_tokens(wm.notes()[0]))


def test_working_memory_as_message_none_when_empty() -> None:
    wm = WorkingMemory(token_budget=100)
    assert wm.as_message() is None


def test_working_memory_as_message_is_system() -> None:
    wm = WorkingMemory(token_budget=100)
    wm.set_goals(["g"])
    msg = wm.as_message()
    assert msg is not None and msg.role == Role.SYSTEM
