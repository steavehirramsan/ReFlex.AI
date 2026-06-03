# ReFlex.AI — Architecture & Implementation Notes

This document maps the README's architecture onto the actual code so contributors can find
their way around quickly. Every component is backend-agnostic: the heavy/GPU pieces (LLM,
embeddings, ANN index) sit behind small interfaces, with deterministic offline defaults so
the whole system runs and is tested without a GPU, network, or API key.

## Data flow for one turn

`Agent.turn()` → `Orchestrator.handle()` runs this pipeline ([orchestrator.py](../src/reflex/core/orchestrator.py)):

1. **Retrieve** — `MemoryManager.retrieve()` embeds the input and searches every durable
   tier, then fuses scores with a recency prior into one ranked `RetrievalBundle`.
2. **Assemble** — `Policy.build_messages()` composes persona + working memory + retrieved
   context + recent conversation into the prompt.
3. **Generate** — the configured `LLMClient` produces a candidate response.
4. **Verify** — `IntegrityGuard.check()` scores grounding and consistency. Per
   `integrity.on_violation`, a blocked draft is `revise`d (re-prompted with the flags),
   `flag`ged, or `raise`d.
5. **Persist + Reflect** — the exchange is written to episodic memory; `ReflectionEngine`
   distils durable facts and folds integrity findings into corrections.
6. **Compact** — `MemoryManager.maybe_compact()` summarises overflowing tiers into the cold
   archive to keep resident memory bounded.

## Module map

| README concept | Code |
|---|---|
| Orchestrator (routing + policy) | [`core/orchestrator.py`](../src/reflex/core/orchestrator.py), [`core/policy.py`](../src/reflex/core/policy.py) |
| Working memory / short-term buffer | [`memory/short_term.py`](../src/reflex/memory/short_term.py) |
| Episodic store (*what happened*) | [`memory/episodic.py`](../src/reflex/memory/episodic.py) |
| Semantic store (*what is true*) | [`memory/semantic.py`](../src/reflex/memory/semantic.py) |
| Compressed archive (cold recall) | [`memory/archive.py`](../src/reflex/memory/archive.py) |
| Tier coordination + retrieval + compaction | [`memory/manager.py`](../src/reflex/memory/manager.py) |
| Reflection (self-correction loop) | [`reflection/engine.py`](../src/reflex/reflection/engine.py) |
| Integrity (hallucination & consistency) | [`integrity/guard.py`](../src/reflex/integrity/guard.py) |
| Long-running runtime | [`runtime/agent.py`](../src/reflex/runtime/agent.py) |
| Evaluation harness | [`eval/harness.py`](../src/reflex/eval/harness.py) |

## Memory tiers

| Tier | Volatility | Backing | Key behaviour |
|---|---|---|---|
| Short-term buffer | volatile | in-process deque | bounded FIFO of raw recent events |
| Working memory | volatile | in-process | token-budgeted goals + notes for the prompt |
| Episodic | durable | SQLite + vector index | append-only event log, searchable by similarity |
| Semantic | durable | SQLite + vector index | distilled facts with supersession (`valid`/`superseded_by`) |
| Archive | durable (cold) | SQLite + vector index | compressed summaries of compacted history |

Every durable store rebuilds its in-memory vector index from persisted embeddings on
construction, so a process restart recovers the full searchable state.

## Retrieval scoring

For a query the manager gathers top-k hits from each durable tier, drops anything below
`min_score`, then ranks by:

```
fused = (1 - recency_weight) · similarity + recency_weight · exp(-age / half_life)
```

so relevance dominates while recency breaks ties — and a recency boost can never resurrect
an irrelevant memory.

## Integrity checks

`IntegrityGuard` is deterministic and algorithmic (no second LLM call on the hot path):

* **fabricated_memory** — the reply claims a memory (“you said…”) absent from retrieved context.
* **low_support** — a factual claim is weakly grounded in retrieved memory.
* **factual_drift** — a claim closely matches an established fact but flips its polarity.
* **inconsistent_output** — the reply contradicts itself across sentences.

## Determinism & testing

The default `mock` LLM and `hashing` embedder are pure functions of their inputs, so the
105-test suite — including the end-to-end agent and the retention benchmark — is fully
reproducible offline. See [`tests/`](../tests/). Swap in real backends via config to run the
identical pipeline on AMD Instinct hardware.

## Extending

* **New LLM backend** — implement `LLMClient` ([`llm/base.py`](../src/reflex/llm/base.py)) and wire it in `llm/__init__.py:build_llm`.
* **New embedder** — implement `Embedder` and register in `embeddings/__init__.py:build_embedder`.
* **New vector backend** — implement `VectorIndex` and register in `memory/vector_index.py:build_vector_index`.
* **Postgres** — the `Database` surface is intentionally tiny; add an adapter behind the same store classes.
