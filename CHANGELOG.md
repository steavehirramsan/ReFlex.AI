# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-06-03

Initial release: a runnable, fully-tested implementation of the ReFlex cognitive runtime.

### Added
- **Tiered memory subsystem** — short-term buffer and working memory (volatile); durable
  episodic, semantic, and compressed-archive stores backed by SQLite with rebuildable
  vector indexes.
- **Memory manager** — unified cross-tier retrieval with recency-weighted score fusion, plus
  autonomous compaction of overflowing tiers into the cold archive.
- **Orchestrator + policy** — the cognition loop: retrieve → assemble → generate → verify →
  persist → reflect → compact.
- **Reflection engine** — closed self-correction loop that distils durable facts (heuristic +
  optional LLM augmentation) and folds integrity findings into corrections.
- **Integrity layer** — deterministic guard flagging fabricated memory, low support, factual
  drift, and self-inconsistency, with `flag` / `revise` / `raise` policies.
- **Pluggable backends** — `LLMClient` (deterministic offline mock + OpenAI-compatible client
  for vLLM/SGLang), `Embedder` (hashing default + optional sentence-transformers), and
  `VectorIndex` (numpy default + optional FAISS).
- **Config system** — typed, validated config with `defaults < YAML < env < overrides`
  precedence and `REFLEX_*` environment overrides.
- **CLI** — `reflex run | chat | inspect | eval | config | version`.
- **Evaluation harness** — reproducible memory-retention benchmark.
- **Tooling** — 105-test pytest suite, `ruff`, `mypy --strict`, GitHub Actions CI, packaging
  via Hatchling, docs, and runnable example.
