"""Reflection engine — the self-correction step of the cognition loop.

After the agent responds, reflection runs the *Observe → Reflect → Detect → Correct → Write*
arc described in the README: it distils durable facts out of the exchange, folds integrity
findings into explicit corrections, and writes both back to memory so the next turn starts
from an improved state.

Fact distillation has two layers:

* a deterministic **heuristic extractor** that captures explicit, stable assertions
  ("remember that…", "my X is Y", "A is B") — this runs offline and makes the loop testable;
* an optional **LLM augmentation** pass that asks the model for additional ``FACT:`` lines,
  used only when a real model is configured.

Keeping a working deterministic core means the self-correction loop is exercised by the test
suite without any model in the loop.
"""

from __future__ import annotations

import re

from ..config import ReflectionConfig
from ..llm.base import LLMClient
from ..logging import get_logger
from ..memory.manager import MemoryManager
from ..types import IntegrityReport, Message, ReflectionResult, Role

log = get_logger(__name__)

# Heuristic patterns for stable, declarative facts worth persisting.
_REMEMBER = re.compile(r"\b(?:remember|note|keep in mind)(?:\s+that)?\s+(.+)", re.IGNORECASE)
_MY_ATTR = re.compile(r"\bmy\s+([a-z][\w \-]{1,40}?)\s+(?:is|are|=)\s+(.+)", re.IGNORECASE)
_IS_A = re.compile(r"^([A-Z][\w .\-]{1,60}?)\s+(?:is|are)\s+(.+)$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HAS_DIGIT = re.compile(r"\d")


def _is_salient(sentence: str) -> bool:
    """A declarative sentence is *salient* (worth remembering) when it carries a concrete
    value — a number/code or a proper noun — rather than being vague chit-chat.

    This is the precision lever that lets the offline extractor promote
    "the staging cluster lives in region eu-west-2" to a durable fact while ignoring
    "the afternoon went smoothly". Real deployments additionally use LLM extraction.
    """
    if sentence.endswith("?"):
        return False
    words = sentence.split()
    if not (4 <= len(words) <= 40):
        return False
    if _HAS_DIGIT.search(sentence):
        return True
    # An interior capitalised, alphabetic token is a proper-noun signal.
    return any(
        len(tok := w.strip(".,;:'\"()")) >= 3 and tok[0].isupper() and tok.isalpha()
        for w in words[1:]
    )


_FACT_PROMPT = (
    "From the exchange below, extract durable facts worth remembering long-term. "
    "Output zero or more lines, each starting with 'FACT: '. Only include stable, "
    "verifiable statements — never speculation or transient chit-chat.\n\n"
    "USER: {user}\nASSISTANT: {assistant}"
)
_FACT_LINE = re.compile(r"^\s*FACT:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


class ReflectionEngine:
    """Runs reflection after each turn and writes corrections to durable memory."""

    def __init__(self, config: ReflectionConfig, llm: LLMClient | None = None) -> None:
        self.config = config
        self._llm = llm

    async def reflect(
        self,
        *,
        session_id: str,
        user_input: str,
        response: str,
        integrity: IntegrityReport,
        memory: MemoryManager,
        source_event_ids: list[str] | None = None,
    ) -> ReflectionResult:
        if not self.config.enabled:
            return ReflectionResult()

        facts = self._heuristic_facts(user_input)
        facts.update(await self._llm_facts(user_input, response))

        corrections = self._corrections_from_integrity(integrity)
        drift = integrity.blocking(0.5) or any(
            f.kind.value in ("factual_drift", "inconsistent_output") for f in integrity.flags
        )

        importance = self._importance(facts, integrity)
        stored: list[str] = []
        if importance >= 0 and facts:
            for statement in sorted(facts):
                confidence = 0.85 if statement.lower().startswith("user") else 0.7
                if confidence >= self.config.min_importance_to_store:
                    memory.add_fact(
                        statement,
                        confidence=confidence,
                        source_event_ids=source_event_ids or [],
                    )
                    stored.append(statement)

        summary = self._summary(stored, corrections, drift)
        memory.record_event(
            session_id,
            summary,
            role=Role.ASSISTANT,
            kind="reflection",
            importance=importance,
            metadata={"drift": drift, "facts_stored": len(stored)},
        )

        return ReflectionResult(
            summary=summary,
            drift_detected=drift,
            corrections=corrections,
            new_facts=stored,
            importance=importance,
        )

    # -- fact extraction ---------------------------------------------------

    def _heuristic_facts(self, text: str) -> set[str]:
        facts: set[str] = set()
        for sentence in _SENTENCE_SPLIT.split(text.strip()):
            sentence = sentence.strip().rstrip(".")
            if not sentence:
                continue
            if m := _REMEMBER.search(sentence):
                facts.add(m.group(1).strip().rstrip("."))
            elif m := _MY_ATTR.search(sentence):
                attr, val = m.group(1).strip(), m.group(2).strip().rstrip(".")
                facts.add(f"User's {attr} is {val}")
            elif _IS_A.match(sentence) or _is_salient(sentence):
                facts.add(sentence)
        return {f for f in facts if len(f) >= 3}

    async def _llm_facts(self, user_input: str, response: str) -> set[str]:
        if self._llm is None:
            return set()
        try:
            prompt = _FACT_PROMPT.format(user=user_input, assistant=response)
            out = await self._llm.complete(
                [Message(role=Role.USER, content=prompt)], temperature=0.0
            )
        except Exception as exc:
            log.warning("LLM fact extraction failed, using heuristics only: %s", exc)
            return set()
        return {m.group(1).strip().rstrip(".") for m in _FACT_LINE.finditer(out) if m.group(1)}

    # -- corrections & scoring --------------------------------------------

    @staticmethod
    def _corrections_from_integrity(integrity: IntegrityReport) -> list[str]:
        corrections = []
        for flag in integrity.flags:
            corrections.append(f"[{flag.kind.value}] {flag.message}")
        return corrections

    @staticmethod
    def _importance(facts: set[str], integrity: IntegrityReport) -> float:
        base = 0.4
        if facts:
            base += 0.3
        base += 0.3 * integrity.max_severity  # drift/issues make the turn more notable
        return min(1.0, base)

    @staticmethod
    def _summary(stored: list[str], corrections: list[str], drift: bool) -> str:
        bits = []
        if stored:
            bits.append(f"learned {len(stored)} fact(s)")
        if corrections:
            bits.append(f"{len(corrections)} integrity correction(s)")
        if drift:
            bits.append("drift detected")
        return "Reflection: " + (", ".join(bits) if bits else "no new durable state")
