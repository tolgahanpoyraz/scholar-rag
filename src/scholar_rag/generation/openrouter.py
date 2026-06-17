from __future__ import annotations

import os
from collections.abc import Iterator

from scholar_rag.generation.base import Generator
from scholar_rag.generation.prompt import SYSTEM_PROMPT, build_user_prompt
from scholar_rag.models import Answer, RetrievedChunk

_DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
_BASE_URL = "https://openrouter.ai/api/v1"

class OpenRouterGenerator(Generator):
    def __init__(
            self,
            model: str = _DEFAULT_MODEL,
            api_key: str | None = None,
            temperature: float = 0.2,
    ) -> None:
        from openai import OpenAI

        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "No API key. Set OPENROUTER_API_KEY in your environment "
                "or pass api_key=..."
            )
        self._client = OpenAI(
            base_url=_BASE_URL,
            api_key=key,
            default_headers={"X-Title": "scholar-rag"},
        )
        self._model = model
        self._temperature = temperature

    def _messages(self, query: str, chunks: list[RetrievedChunk]) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(query, chunks)},
        ]

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> Answer:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=self._messages(query, chunks),
        )
        text = resp.choices[0].message.content or ""
        return Answer(text=text, citations=chunks)

    def generate_stream(self, query: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=self._messages(query, chunks),
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and (delta := chunk.choices[0].delta.content):
                yield delta