"""Microphone + energy VAD + speaker, via sounddevice (PortAudio). CPU only.

Use HEADPHONES: without echo cancellation the agent hears itself and barges in on
its own voice. (Next step: WebRTC in the browser gives you echo cancellation for free.)
"""
from __future__ import annotations

import asyncio
from typing import Callable

import numpy as np

from .base import AudioChunk, AudioOut

MIC_RATE = 16000
FRAME_MS = 20
FRAME = MIC_RATE * FRAME_MS // 1000


class Mic:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[tuple[float, bytes]] = asyncio.Queue()
        self.stream = None

    def start(self) -> None:
        import sounddevice as sd
        loop = asyncio.get_running_loop()

        def cb(indata, frames, t, status):
            item = (loop.time() * 1000.0, bytes(indata))  # loop clock is monotonic & thread-safe to read
            loop.call_soon_threadsafe(self.queue.put_nowait, item)

        self.stream = sd.RawInputStream(samplerate=MIC_RATE, channels=1, dtype="int16",
                                        blocksize=FRAME, callback=cb)
        self.stream.start()

    def stop(self) -> None:
        if self.stream:
            self.stream.stop()
            self.stream.close()


class EnergyVAD:
    """Adaptive-threshold energy VAD. Swap for Silero VAD (CPU, ~1 ms/frame) for noisy rooms."""

    def __init__(self, on_start: Callable[[float], None], on_stop: Callable[[float], None],
                 start_frames: int = 3, stop_ms: float = 150, calib_frames: int = 25):
        self.on_start, self.on_stop = on_start, on_stop
        self.start_frames, self.stop_frames = start_frames, int(stop_ms / FRAME_MS)
        self.calib_frames = calib_frames
        self.noise: list[float] = []
        self.threshold = 600.0
        self.speaking = False
        self.voiced_run = 0
        self.silent_run = 0
        self.first_voiced_t = 0.0
        self.first_silent_t = 0.0

    def process(self, t_ms: float, frame: bytes) -> None:
        x = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        if len(self.noise) < self.calib_frames:
            self.noise.append(rms)
            if len(self.noise) == self.calib_frames:
                self.threshold = max(500.0, 3.0 * float(np.median(self.noise)))
                print(f"[vad] calibrated, threshold={self.threshold:.0f}")
            return
        voiced = rms > self.threshold
        if voiced:
            if self.voiced_run == 0:
                self.first_voiced_t = t_ms - FRAME_MS
            self.voiced_run += 1
            self.silent_run = 0
            if not self.speaking and self.voiced_run >= self.start_frames:
                self.speaking = True
                self.on_start(self.first_voiced_t)
        else:
            if self.silent_run == 0:
                self.first_silent_t = t_ms - FRAME_MS
            self.silent_run += 1
            self.voiced_run = 0
            if self.speaking and self.silent_run >= self.stop_frames:
                self.speaking = False
                self.on_stop(self.first_silent_t)


class SpeakerOut(AudioOut):
    def __init__(self, rate: int = 24000):
        import sounddevice as sd
        self.stream = sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16", latency="low")
        self.stream.start()
        self.gain = 1.0

    async def play(self, chunk: AudioChunk) -> None:
        data = chunk.data
        if self.gain != 1.0:
            data = (np.frombuffer(data, dtype=np.int16) * self.gain).astype(np.int16).tobytes()
        await asyncio.to_thread(self.stream.write, data)

    def duck(self, on: bool) -> None:
        self.gain = 0.2 if on else 1.0

    def stop(self) -> None:
        try:  # drop whatever is buffered in the device right now
            self.stream.abort()
            self.stream.start()
        except Exception:
            pass

    def close(self) -> None:
        self.stream.close()
