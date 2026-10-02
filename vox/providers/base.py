from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class AudioChunk:
    data: bytes
    duration_ms: float


class LLM:
    name = "llm"

    async def choose_tool(self, messages: list[dict]) -> str:
        """Function-calling round trip: returns a tool name or 'none'."""
        raise NotImplementedError

    def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream the reply text in small chunks."""
        raise NotImplementedError


class TTS:
    name = "tts"

    def synth(self, text: str) -> AsyncIterator[AudioChunk]:
        raise NotImplementedError


class AudioOut:
    async def play(self, chunk: AudioChunk) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        """Flush anything buffered (barge-in)."""

    def duck(self, on: bool) -> None:
        """Lower (on=True) or restore (on=False) playback volume."""
