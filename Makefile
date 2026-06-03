.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install install-all lint format type test cov check run eval clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with dev dependencies (editable)
	$(PY) -m pip install -e ".[dev]"

install-all:  ## Install with all optional backends (faiss, embeddings, postgres, serve)
	$(PY) -m pip install -e ".[dev,all]"

lint:  ## Lint and check formatting
	ruff check src tests
	ruff format --check src tests

format:  ## Auto-format the codebase
	ruff format src tests
	ruff check --fix src tests

type:  ## Type-check with mypy (strict)
	mypy

test:  ## Run the test suite
	pytest

cov:  ## Run tests with coverage report
	pytest --cov=reflex --cov-report=term-missing

check: lint type test  ## Run the full quality gate (lint + type + test)

run:  ## Launch the interactive REPL (offline mock by default)
	reflex run

eval:  ## Run the memory-retention benchmark
	reflex eval

clean:  ## Remove caches and build/runtime artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist *.egg-info
	rm -f reflex_memory.db *.db-wal *.db-shm
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
