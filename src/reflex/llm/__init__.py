"""LLM backends and the factory that builds one from config."""

from __future__ import annotations

from ..config import LLMConfig
from .base import LLMClient
from .mock import MockLLM
from .openai_compatible import OpenAICompatibleLLM

__all__ = ["LLMClient", "MockLLM", "OpenAICompatibleLLM", "build_llm"]


def build_llm(config: LLMConfig) -> LLMClient:
    """Instantiate the configured LLM backend."""
    if config.provider == "mock":
        return MockLLM()
    if config.provider == "openai":
        return OpenAICompatibleLLM(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
            max_retries=config.max_retries,
        )
    raise ValueError(f"Unknown LLM provider: {config.provider!r}")  # pragma: no cover
