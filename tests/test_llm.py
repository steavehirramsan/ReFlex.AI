from __future__ import annotations

import httpx
import pytest

from reflex.config import LLMConfig
from reflex.errors import LLMError
from reflex.llm import build_llm
from reflex.llm.mock import MockLLM
from reflex.llm.openai_compatible import OpenAICompatibleLLM
from reflex.types import Message, Role


async def test_mock_scripted_returns_in_order() -> None:
    llm = MockLLM(scripted=["first", "second"])
    assert await llm.complete([Message(role=Role.USER, content="x")]) == "first"
    assert await llm.complete([Message(role=Role.USER, content="x")]) == "second"


async def test_mock_grounded_quotes_context() -> None:
    llm = MockLLM()
    messages = [
        Message(role=Role.SYSTEM, content="[semantic:fact_1 score=0.90] the sky is blue"),
        Message(role=Role.USER, content="what color is the sky?"),
    ]
    out = await llm.complete(messages)
    assert "sky is blue" in out


async def test_mock_no_context_admits_ignorance() -> None:
    llm = MockLLM()
    out = await llm.complete([Message(role=Role.USER, content="what is my name?")])
    assert "don't have anything in memory" in out.lower()


def test_build_llm_mock() -> None:
    assert isinstance(build_llm(LLMConfig(provider="mock")), MockLLM)


def test_build_llm_openai() -> None:
    llm = build_llm(LLMConfig(provider="openai", model="m", base_url="http://x/v1"))
    assert isinstance(llm, OpenAICompatibleLLM)


async def test_openai_client_parses_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "hi there"}}]},
        )

    llm = OpenAICompatibleLLM(model="m", base_url="http://test/v1")
    llm._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test/v1"
    )
    out = await llm.complete([Message(role=Role.USER, content="hello")])
    assert out == "hi there"
    await llm.aclose()


async def test_openai_client_4xx_raises_without_retry() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    llm = OpenAICompatibleLLM(model="m", base_url="http://test/v1", max_retries=3)
    llm._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test/v1"
    )
    with pytest.raises(LLMError):
        await llm.complete([Message(role=Role.USER, content="hello")])
    assert calls["n"] == 1  # 4xx is not retried
    await llm.aclose()


async def test_openai_client_malformed_response_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    llm = OpenAICompatibleLLM(model="m", base_url="http://test/v1")
    llm._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test/v1"
    )
    with pytest.raises(LLMError):
        await llm.complete([Message(role=Role.USER, content="hello")])
    await llm.aclose()
