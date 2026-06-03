"""Integrity layer — consistency and hallucination guard.

Every candidate response is checked *before* it is trusted or persisted. The guard is
algorithmic and deterministic (no second LLM call on the hot path), which keeps it fast,
cheap, and testable. It raises four families of flag from the README:

* **fabricated_memory** — the response asserts a memory ("you said…", "earlier we…") that
  has no sufficiently-similar record in retrieved context.
* **low_support** — a factual claim is poorly grounded in any retrieved memory.
* **factual_drift** — a claim closely matches an established valid fact but contradicts its
  polarity (asserts the negation of something believed true).
* **inconsistent_output** — the response contradicts itself across sentences.

Flags carry a severity in ``[0, 1]``; the orchestrator decides what to do with them based on
``integrity.on_violation`` and ``block_threshold``.
"""

from __future__ import annotations

import re

import numpy as np

from ..config import IntegrityConfig
from ..embeddings.base import Embedder
from ..memory.manager import MemoryManager
from ..types import Flag, FlagKind, IntegrityReport, RetrievalBundle

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MEMORY_REF = re.compile(
    r"\b(you (?:said|told|mentioned)|earlier|previously|we discussed|"
    r"as (?:you|i) (?:said|mentioned)|last time|i recall|you asked)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(?:not|no|never|n't|cannot|can't|won't|isn't|aren't|no longer)\b", re.IGNORECASE
)
_HEDGE = re.compile(
    r"\b(?:i think|maybe|perhaps|might|possibly|i'?m not sure|i don'?t (?:know|have))\b",
    re.IGNORECASE,
)


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity matrix between row-vectors of ``a`` and ``b``."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    an = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    bn = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
    return np.asarray(an @ bn.T, dtype=np.float32)


class IntegrityGuard:
    """Checks a candidate response for grounding and consistency."""

    def __init__(self, config: IntegrityConfig, embedder: Embedder) -> None:
        self.config = config
        self._embedder = embedder

    def check(
        self,
        response: str,
        retrieval: RetrievalBundle,
        memory: MemoryManager,
    ) -> IntegrityReport:
        if not self.config.enabled:
            return IntegrityReport()

        claims = self._claim_sentences(response)
        if not claims:
            return IntegrityReport()

        flags: list[Flag] = []
        claim_vecs = self._embedder.embed(claims)

        flags.extend(self._check_grounding(claims, claim_vecs, retrieval))
        flags.extend(self._check_contradiction(claims, claim_vecs, memory))
        flags.extend(self._check_self_consistency(claims, claim_vecs))

        score = 1.0 - max((f.severity for f in flags), default=0.0)
        return IntegrityReport(score=round(score, 4), flags=flags)

    # -- claim detection ---------------------------------------------------

    @staticmethod
    def _claim_sentences(text: str) -> list[str]:
        out = []
        for s in _SENTENCE_SPLIT.split(text.strip()):
            s = s.strip()
            if len(s.split()) < 4 or s.endswith("?"):
                continue  # too short or a question — not a factual claim
            if _HEDGE.search(s):
                continue  # explicitly hedged statements are not asserted as fact
            out.append(s)
        return out

    # -- checks ------------------------------------------------------------

    def _check_grounding(
        self, claims: list[str], claim_vecs: np.ndarray, retrieval: RetrievalBundle
    ) -> list[Flag]:
        flags: list[Flag] = []
        context = [h.content for h in retrieval.hits]
        ctx_vecs = self._embedder.embed(context) if context else np.zeros((0, claim_vecs.shape[1]))
        sims = _cosine_matrix(claim_vecs, ctx_vecs)

        for i, claim in enumerate(claims):
            best = float(sims[i].max()) if sims.shape[1] else 0.0
            references_memory = bool(_MEMORY_REF.search(claim))
            if references_memory and best < self.config.support_threshold:
                # The response claims a memory that isn't in the retrieved record.
                sev = self._severity(best, ceiling=0.9)
                flags.append(
                    Flag(
                        kind=FlagKind.FABRICATED_MEMORY,
                        severity=sev,
                        message=(
                            "Response references prior interaction not found in memory: "
                            f"“{claim[:120]}”"
                        ),
                        evidence=[f"best_support={best:.3f}"],
                    )
                )
            elif context and references_memory is False and best < self.config.support_threshold:
                flags.append(
                    Flag(
                        kind=FlagKind.LOW_SUPPORT,
                        severity=self._severity(best, ceiling=0.5),
                        message=f"Claim weakly grounded in retrieved memory: “{claim[:120]}”",
                        evidence=[f"best_support={best:.3f}"],
                    )
                )
        return flags

    def _check_contradiction(
        self, claims: list[str], claim_vecs: np.ndarray, memory: MemoryManager
    ) -> list[Flag]:
        facts = memory.semantic.all_valid(limit=500)
        if not facts:
            return []
        # Re-embed with the guard's own embedder so the matrix dims always agree, regardless
        # of which embedder originally populated the fact store.
        fact_vecs = self._embedder.embed([f.statement for f in facts])
        sims = _cosine_matrix(claim_vecs, fact_vecs)
        flags: list[Flag] = []
        for i, claim in enumerate(claims):
            j = int(sims[i].argmax())
            sim = float(sims[i][j])
            if sim < 0.55:
                continue  # not talking about the same thing
            fact = facts[j]
            if _polarity(claim) != _polarity(fact.statement):
                flags.append(
                    Flag(
                        kind=FlagKind.FACTUAL_DRIFT,
                        severity=min(1.0, sim),
                        message=(
                            f"Claim contradicts established fact (sim={sim:.2f}): "
                            f"“{claim[:100]}” vs “{fact.statement[:100]}”"
                        ),
                        evidence=[fact.id],
                    )
                )
        return flags

    def _check_self_consistency(self, claims: list[str], claim_vecs: np.ndarray) -> list[Flag]:
        flags: list[Flag] = []
        sims = _cosine_matrix(claim_vecs, claim_vecs)
        n = len(claims)
        for i in range(n):
            for k in range(i + 1, n):
                if sims[i][k] > 0.6 and _polarity(claims[i]) != _polarity(claims[k]):
                    flags.append(
                        Flag(
                            kind=FlagKind.INCONSISTENT_OUTPUT,
                            severity=float(sims[i][k]),
                            message=(
                                "Response contradicts itself: "
                                f"“{claims[i][:80]}” vs “{claims[k][:80]}”"
                            ),
                        )
                    )
        return flags

    @staticmethod
    def _severity(support: float, *, ceiling: float) -> float:
        """Map a (low) support score to a severity, capped at ``ceiling``."""
        return round(min(ceiling, max(0.0, 1.0 - support * 5)), 4)


def _polarity(text: str) -> bool:
    """Coarse truth polarity: ``False`` if the sentence is negated, else ``True``."""
    return _NEGATION.search(text) is None
