"""OpenAI-compatible chat LLM (Groq, OpenAI, Together, a local vLLM/Ollama server...).

Groq is a good default for this project: free tier, and very low time-to-first-token.
"""
from __future__ import annotations

import json
import os
from typing import AsyncIterator

import httpx

from ..tools import openai_tool_specs
from .base import LLM


class OpenAICompatLLM(LLM):
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key or os.environ["LLM_API_KEY"]
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
        self.name = f"llm:{self.model}"
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0),
                                        headers={"Authorization": f"Bearer {self.api_key}"})

    async def choose_tool(self, messages: list[dict]) -> str:
        r = await self.client.post(f"{self.base_url}/chat/completions", json={
            "model": self.model, "messages": messages, "tools": openai_tool_specs(),
            "tool_choice": "auto", "temperature": 0, "max_tokens": 60})
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        return calls[0]["function"]["name"] if calls else "none"

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        body = {"model": self.model, "messages": messages, "stream": True, "temperature": 0.4, "max_tokens": 120}
        async with self.client.stream("POST", f"{self.base_url}/chat/completions", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta

    async def aclose(self) -> None:
        await self.client.aclose()
