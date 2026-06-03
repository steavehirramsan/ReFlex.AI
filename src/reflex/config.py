"""Configuration model and loader for ReFlex.

Configuration is a tree of typed, validated :class:`pydantic.BaseModel` sections. Values
are resolved with an explicit, predictable precedence::

    code defaults  <  YAML file  <  REFLEX_* environment variables  <  explicit overrides

Environment variables use the ``REFLEX_`` prefix and ``__`` to descend into nested
sections, e.g. ``REFLEX_LLM__MODEL=meta-llama/Llama-3.1-8B-Instruct`` or
``REFLEX_MEMORY__RETRIEVAL__TOTAL_K=12``. This keeps secrets (API keys, DSNs) out of
checked-in YAML while letting the file carry structural config.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from .errors import ConfigError

ENV_PREFIX = "REFLEX_"
ENV_NESTED_DELIM = "__"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    name: str = "reflex"
    persona: str = (
        "You are ReFlex, a careful, long-running assistant with durable memory. "
        "You ground answers in retrieved memory, admit uncertainty, and never invent history."
    )
    goals: list[str] = Field(default_factory=list)
    session_id: str | None = None


class LLMConfig(BaseModel):
    # 'mock' is deterministic and offline (default so the system runs anywhere).
    # 'openai' speaks the OpenAI chat API, which vLLM and SGLang both expose on ROCm.
    provider: Literal["mock", "openai"] = "mock"
    model: str = "reflex-mock"
    base_url: str | None = None  # e.g. http://localhost:8000/v1 for a local vLLM server
    api_key: str | None = None
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)


class EmbeddingConfig(BaseModel):
    # 'hashing' is a deterministic, dependency-free embedder for offline use and tests.
    # 'sentence_transformers' loads a real model (extra: reflex-memory[embeddings]).
    provider: Literal["hashing", "sentence_transformers"] = "hashing"
    model: str = "hashing-256"
    dim: int = Field(default=256, gt=0)
    normalize: bool = True


class VectorConfig(BaseModel):
    backend: Literal["numpy", "faiss"] = "numpy"
    metric: Literal["cosine", "ip", "l2"] = "cosine"


class RetrievalConfig(BaseModel):
    episodic_k: int = Field(default=6, ge=0)
    semantic_k: int = Field(default=6, ge=0)
    archive_k: int = Field(default=3, ge=0)
    total_k: int = Field(default=10, ge=1)
    min_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    recency_half_life_s: float = Field(default=86_400.0, gt=0)  # 1 day
    recency_weight: float = Field(default=0.15, ge=0.0, le=1.0)


class CompactionConfig(BaseModel):
    enabled: bool = True
    episodic_threshold: int = Field(default=200, gt=0)  # events before compaction
    semantic_threshold: int = Field(default=500, gt=0)  # facts before compaction
    batch_size: int = Field(default=50, gt=0)
    keep_recent: int = Field(default=50, ge=0)  # most-recent events spared from compaction


class MemoryConfig(BaseModel):
    db_path: str = "reflex_memory.db"
    short_term_capacity: int = Field(default=40, gt=0)
    working_token_budget: int = Field(default=2048, gt=0)
    promotion_importance: float = Field(default=0.6, ge=0.0, le=1.0)
    vector: VectorConfig = Field(default_factory=VectorConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)


class ReflectionConfig(BaseModel):
    enabled: bool = True
    every_n_turns: int = Field(default=1, ge=1)
    min_importance_to_store: float = Field(default=0.55, ge=0.0, le=1.0)


class IntegrityConfig(BaseModel):
    enabled: bool = True
    support_threshold: float = Field(default=0.18, ge=0.0, le=1.0)
    block_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    on_violation: Literal["flag", "revise", "raise"] = "revise"
    max_revisions: int = Field(default=1, ge=0)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    rich: bool = True


class ReflexConfig(BaseModel):
    """Root configuration object."""

    agent: AgentConfig = Field(default_factory=AgentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    integrity: IntegrityConfig = Field(default_factory=IntegrityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # -- loading -----------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        env: bool = True,
        overrides: dict[str, Any] | None = None,
    ) -> ReflexConfig:
        """Build a config from defaults, an optional YAML file, env vars, and overrides.

        Raises
        ------
        ConfigError
            If the YAML file is missing/malformed or the merged config fails validation.
        """
        data: dict[str, Any] = {}
        if path is not None:
            data = _deep_merge(data, _read_yaml(path))
        if env:
            data = _deep_merge(data, _env_overrides(os.environ))
        if overrides:
            data = _deep_merge(data, overrides)
        try:
            return cls.model_validate(data)
        except ValidationError as exc:  # pragma: no cover - exercised via tests
            raise ConfigError(f"Invalid ReFlex configuration:\n{exc}") from exc

    def to_yaml(self) -> str:
        """Serialise the resolved config back to YAML (secrets included — handle with care)."""
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")
    try:
        loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML config {p}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Top-level YAML config must be a mapping, got {type(loaded).__name__}")
    return loaded


def _env_overrides(environ: Mapping[str, str]) -> dict[str, Any]:
    """Translate ``REFLEX_SECTION__KEY=value`` env vars into a nested dict.

    Values are parsed as YAML scalars so ``true``/``42``/``0.5`` get native types while
    arbitrary strings pass through untouched.
    """
    out: dict[str, Any] = {}
    for raw_key, raw_val in environ.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        path = raw_key[len(ENV_PREFIX) :].lower().split(ENV_NESTED_DELIM)
        if not path or not path[0]:
            continue
        cursor = out
        for part in path[:-1]:
            nxt = cursor.setdefault(part, {})
            if not isinstance(nxt, dict):  # collision: a scalar already sits here
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[path[-1]] = _coerce_scalar(raw_val)
    return out


def _coerce_scalar(value: str) -> Any:
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        return value
    # Only adopt scalar coercions; never let a stray "{...}" become a dict here.
    return parsed if isinstance(parsed, (int, float, bool)) or parsed is None else value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either argument."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
