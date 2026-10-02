"""Deepgram streaming STT (WebSocket) and Aura TTS (HTTP streaming).

One Deepgram key covers both. Docs: developers.deepgram.com
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncIterator, Callable

import httpx

from ..clock import now_ms
from ..orchestrator import Word
from .base import TTS, AudioChunk

STT_URL = ("wss://api.deepgram.com/v1/listen?model={model}&encoding=linear16&sample_rate=16000&channels=1"
           "&interim_results=true&punctuate=true&smart_format=true")
TTS_URL = "https://api.deepgram.com/v1/speak?model={model}&encoding=linear16&sample_rate=24000&container=none"
TTS_RATE = 24000


class DeepgramSTT:
    """Send 16 kHz int16 mono frames in, get word hypotheses (with timestamps) out."""

    def __init__(self, on_words: Callable[[list[Word]], None], api_key: str | None = None, model: str | None = None):
        self.key = api_key or os.environ["DEEPGRAM_API_KEY"]
        self.url = STT_URL.format(model=model or os.environ.get("STT_MODEL", "nova-3"))
        self.on_words = on_words
        self.final: list[Word] = []
        self.interim: list[Word] = []
        self.t0: float | None = None  # loop time (ms) of the first audio frame we sent
        self.ws = None
        self._reader: asyncio.Task | None = None

    async def connect(self) -> None:
        import websockets
        headers = {"Authorization": f"Token {self.key}"}
        try:  # websockets >= 14
            self.ws = await websockets.connect(self.url, additional_headers=headers, max_size=None)
        except TypeError:  # older versions
            self.ws = await websockets.connect(self.url, extra_headers=headers, max_size=None)
        self._reader = asyncio.create_task(self._read())

    async def send(self, frame: bytes) -> None:
        if self.t0 is None:
            self.t0 = now_ms()
        await self.ws.send(frame)

    async def _read(self) -> None:
        async for raw in self.ws:
            msg = json.loads(raw)
            if msg.get("type") != "Results" or self.t0 is None:
                continue
            alt = msg["channel"]["alternatives"][0]
            words = [Word(w.get("punctuated_word") or w["word"], self.t0 + w["start"] * 1000, self.t0 + w["end"] * 1000)
                     for w in alt.get("words", [])]
            if msg.get("is_final"):
                self.final += words
                self.interim = []
            else:
                self.interim = words
            self.final = self.final[-400:]
            self.on_words(self.final + self.interim)

    async def close(self) -> None:
        if self.ws:
            try:
                await self.ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
            await self.ws.close()
        if self._reader:
            self._reader.cancel()


class DeepgramTTS(TTS):
    def __init__(self, api_key: str | None = None, model: str | None = None, chunk_ms: float = 100):
        self.key = api_key or os.environ["DEEPGRAM_API_KEY"]
        self.url = TTS_URL.format(model=model or os.environ.get("TTS_MODEL", "aura-2-thalia-en"))
        self.name = "deepgram-aura"
        self.chunk_bytes = int(TTS_RATE * 2 * chunk_ms / 1000)
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0),
                                        headers={"Authorization": f"Token {self.key}"})

    async def synth(self, text: str) -> AsyncIterator[AudioChunk]:
        buf = b""
        async with self.client.stream("POST", self.url, json={"text": text}) as r:
            r.raise_for_status()
            async for b in r.aiter_bytes():
                buf += b
                while len(buf) >= self.chunk_bytes:
                    data, buf = buf[:self.chunk_bytes], buf[self.chunk_bytes:]
                    yield AudioChunk(data, len(data) / 2 / TTS_RATE * 1000)
        if len(buf) >= 2:
            buf = buf[: len(buf) // 2 * 2]
            yield AudioChunk(buf, len(buf) / 2 / TTS_RATE * 1000)

    async def aclose(self) -> None:
        await self.client.aclose()
