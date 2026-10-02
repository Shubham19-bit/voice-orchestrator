"""Simulated LLM / TTS / speaker with realistic latency distributions.

Defaults are in the range of public measurements for hosted APIs (fast LLM
time-to-first-token ~250-500 ms, streaming TTS first byte ~100-250 ms). Tune them
in `MockProfile` or measure your real providers with `python -m vox.cli probe`.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import AsyncIterator, Callable

from ..clock import sleep_ms
from .base import LLM, TTS, AudioChunk, AudioOut


@dataclass
class MockProfile:
    llm_ttft_p50_ms: float = 380
    llm_tool_call_extra_ms: float = 160     # generating the function-call JSON before the tool can run
    llm_ms_per_word: float = 18
    tts_ttfb_p50_ms: float = 170
    speech_ms_per_word: float = 330
    chunk_ms: float = 200
    jitter_sigma: float = 0.3


GENERIC_REPLIES = [
    "Got it. Is there anything else I can help you with today?",
    "Okay, thank you. Is there anything else you'd like to check?",
    "Sure. Anything else I can do for you?",
]


class MockLLM(LLM):
    name = "mock-llm"

    def __init__(self, rng: random.Random, profile: MockProfile, oracle_tool: Callable[[], str] | None = None):
        self.rng, self.p = rng, profile
        self.oracle_tool = oracle_tool or (lambda: "none")
        self._i = 0

    def _lat(self, p50: float) -> float:
        return p50 * self.rng.lognormvariate(0, self.p.jitter_sigma)

    async def choose_tool(self, messages: list[dict]) -> str:
        # the mock LLM routes perfectly (oracle) — the baseline gets best-case accuracy,
        # it only pays the latency of an extra LLM round trip.
        await sleep_ms(self._lat(self.p.llm_ttft_p50_ms) + self.p.llm_tool_call_extra_ms)
        return self.oracle_tool()

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        text = None
        for m in messages:
            if m["role"] == "system" and m["content"].startswith("Tool result:"):
                text = m["content"].split("Tool result:", 1)[1].strip()
        if not text:
            text = GENERIC_REPLIES[self._i % len(GENERIC_REPLIES)]
            self._i += 1
        await sleep_ms(self._lat(self.p.llm_ttft_p50_ms))
        for w in text.split(" "):
            yield w + " "
            await sleep_ms(self.p.llm_ms_per_word)


class MockTTS(TTS):
    name = "mock-tts"

    def __init__(self, rng: random.Random, profile: MockProfile):
        self.rng, self.p = rng, profile

    async def synth(self, text: str) -> AsyncIterator[AudioChunk]:
        await sleep_ms(self.p.tts_ttfb_p50_ms * self.rng.lognormvariate(0, self.p.jitter_sigma))
        total = max(1, len(re.findall(r"\S+", text))) * self.p.speech_ms_per_word
        n = math.ceil(total / self.p.chunk_ms)
        for i in range(n):
            dur = min(self.p.chunk_ms, total - i * self.p.chunk_ms)
            yield AudioChunk(b"", dur)
            # synthesis runs faster than real time; small gap between chunks
            await sleep_ms(dur * 0.25)


class SimAudioOut(AudioOut):
    async def play(self, chunk: AudioChunk) -> None:
        await sleep_ms(chunk.duration_ms)

    def stop(self) -> None:
        pass
