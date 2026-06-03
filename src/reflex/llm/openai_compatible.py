"""OpenAI-compatible chat backend.

Speaks the ``/v1/chat/completions`` API exposed by OpenAI as well as the ROCm-native
serving stacks named in the README — **vLLM** and **SGLang** both serve this exact schema.
Point ``base_url`` at the server (e.g. ``http://localhost:8000/v1``) and you are running on
AMD Instinct hardware with no code changes elsewhere.
"""

from __future__ import annotations

import asyncio

import httpx

from ..errors import LLMError
from ..logging import get_logger
from ..types import Message
from .base import LLMClient

log = get_logger(__name__)


class OpenAICompatibleLLM(LLMClient):
    """Async client for any OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 1024,
        timeout_s: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens
        self._max_retries = max_retries
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url, headers=headers, timeout=timeout_s
        )

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "temperature": self._default_temperature if temperature is None else temperature,
            "max_tokens": self._default_max_tokens if max_tokens is None else max_tokens,
        }
        if stop:
            payload["stop"] = stop

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                return self._parse(resp.json())
            except httpx.HTTPStatusError as exc:
                # 4xx (except 429) are not worth retrying.
                status = exc.response.status_code
                if status != 429 and 400 <= status < 500:
                    raise LLMError(
                        f"LLM request failed ({status}): {exc.response.text[:500]}"
                    ) from exc
                last_exc = exc
            except (httpx.TransportError, httpx.HTTPError) as exc:
                last_exc = exc

            if attempt < self._max_retries:
                backoff = 0.5 * (2**attempt)
                log.warning("LLM call failed (attempt %d), retrying in %.1fs", attempt + 1, backoff)
                await asyncio.sleep(backoff)

        raise LLMError(f"LLM request failed after {self._max_retries + 1} attempts: {last_exc}")

    @staticmethod
    def _parse(body: dict[str, object]) -> str:
        try:
            choices = body["choices"]
            content = choices[0]["message"]["content"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Malformed LLM response: {body!r}") from exc
        if not isinstance(content, str):
            raise LLMError(f"LLM returned non-string content: {content!r}")
        return content

    async def aclose(self) -> None:
        await self._client.aclose()
