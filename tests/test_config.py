from __future__ import annotations

import pytest

from reflex.config import ReflexConfig, _coerce_scalar, _deep_merge, _env_overrides
from reflex.errors import ConfigError


def test_defaults_are_valid() -> None:
    cfg = ReflexConfig()
    assert cfg.llm.provider == "mock"
    assert cfg.embeddings.dim == 256
    assert cfg.memory.retrieval.total_k >= 1


def test_load_from_yaml(tmp_path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  provider: openai\n  model: my-model\nembeddings:\n  dim: 64\n")
    cfg = ReflexConfig.load(p, env=False)
    assert cfg.llm.provider == "openai"
    assert cfg.llm.model == "my-model"
    assert cfg.embeddings.dim == 64


def test_missing_yaml_raises() -> None:
    with pytest.raises(ConfigError):
        ReflexConfig.load("does-not-exist.yaml", env=False)


def test_malformed_yaml_raises(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ConfigError):
        ReflexConfig.load(p, env=False)


def test_env_overrides_nested() -> None:
    env = {"REFLEX_LLM__MODEL": "env-model", "REFLEX_MEMORY__RETRIEVAL__TOTAL_K": "3"}
    data = _env_overrides(env)
    assert data["llm"]["model"] == "env-model"
    assert data["memory"]["retrieval"]["total_k"] == 3


def test_precedence_env_beats_yaml(tmp_path, monkeypatch) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  model: yaml-model\n")
    monkeypatch.setenv("REFLEX_LLM__MODEL", "env-model")
    cfg = ReflexConfig.load(p, env=True)
    assert cfg.llm.model == "env-model"


def test_overrides_beat_everything(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("REFLEX_LLM__MODEL", "env-model")
    cfg = ReflexConfig.load(env=True, overrides={"llm": {"model": "override-model"}})
    assert cfg.llm.model == "override-model"


def test_coerce_scalar_types() -> None:
    assert _coerce_scalar("42") == 42
    assert _coerce_scalar("0.5") == 0.5
    assert _coerce_scalar("true") is True
    assert _coerce_scalar("hello") == "hello"
    assert _coerce_scalar("{not: a dict}") == "{not: a dict}"


def test_deep_merge_is_pure() -> None:
    base = {"a": {"x": 1, "y": 2}}
    override = {"a": {"y": 3, "z": 4}}
    merged = _deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 3, "z": 4}}
    assert base == {"a": {"x": 1, "y": 2}}  # unchanged


def test_invalid_value_raises_config_error() -> None:
    with pytest.raises(ConfigError):
        ReflexConfig.load(env=False, overrides={"llm": {"temperature": 99}})


def test_roundtrip_yaml() -> None:
    cfg = ReflexConfig()
    text = cfg.to_yaml()
    assert "llm:" in text and "memory:" in text
