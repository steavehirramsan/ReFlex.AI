# Contributing to ReFlex.AI

Thanks for your interest! ReFlex is open research + infrastructure; we value reproducible,
transparent contributions. Open an issue to discuss direction before large PRs.

## Development setup

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional backends for real deployments:

```bash
pip install -e ".[faiss,embeddings]"   # FAISS index + sentence-transformers
```

## The quality gate

Every change must keep these green (CI enforces them):

```bash
make lint        # ruff check + format --check
make type        # mypy --strict
make test        # pytest (offline, deterministic)
# or all at once:
make check
```

Without `make`:

```bash
ruff check src tests && ruff format --check src tests
mypy
pytest --cov=reflex
```

## Conventions

* **Backends stay behind interfaces.** New LLMs/embedders/indexes implement the abstract
  base and register in the relevant `build_*` factory. Don't import optional heavy deps at
  module top level — import them lazily inside the backend so the core stays importable.
* **The core is deterministic and offline-testable.** Memory, retrieval, integrity, and
  reflection must work (and be tested) with the `mock` LLM and `hashing` embedder. The LLM is
  for generation and optional fact augmentation, never a hard dependency of the machinery.
* **Type everything.** `mypy --strict` must pass. Public APIs carry docstrings.
* **Tests for behaviour, not implementation.** Add/extend tests in `tests/`; keep them fast
  and hermetic (in-memory SQLite, fixed seeds).

## Commit / PR

* Keep PRs focused; describe the motivation and the verification you ran.
* Update `CHANGELOG.md` under *Unreleased*.
* If you change config shape, update `configs/*.yaml` and `docs/configuration.md`.
