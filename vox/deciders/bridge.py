"""Use a REAL network decider (Jev / LLM) inside the virtual-time simulator.

Trick: the simulator runs in virtual time. When the orchestrator asks the decider
something, we make the real HTTP call *blocking* on a background thread (virtual
time is frozen while we block), measure its true wall-clock latency, then sleep
that long in virtual time. Result: real answers + real latencies, replayed into a
deterministic simulation. Answers are cached on disk so re-runs are free and
reproducible (delete the cache file to re-measure).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import asdict
from pathlib import Path

from ..clock import sleep_ms
from .base import Decider, OverlapAnalysis, TurnAnalysis


class _BgLoop:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


class SimBridge(Decider):
    def __init__(self, inner: Decider, cache_file: Path | None = None):
        super().__init__()
        self.inner = inner
        self.name = inner.name
        self.cache_file = cache_file
        self.cache: dict[str, dict] = {}
        if cache_file and cache_file.exists():
            self.cache = json.loads(cache_file.read_text())
        self._bg: _BgLoop | None = None

    def _key(self, method: str, *args) -> str:
        model = getattr(self.inner, "model", "")
        return hashlib.sha1(json.dumps([self.inner.name, model, method, *args]).encode()).hexdigest()

    async def _call(self, method: str, cls, *args):
        key = self._key(method, *args)
        rec = self.cache.get(key)
        if rec is None:
            self._bg = self._bg or _BgLoop()
            try:
                res = self._bg.run(getattr(self.inner, method)(*args))
                rec = {"result": asdict(res), "latency_ms": res.latency_ms}
            except Exception as e:  # remember failures too (latency unknown -> assume budget blown)
                rec = {"error": repr(e), "latency_ms": 5000.0}
            self.cache[key] = rec
            self._save()
        self.stats.calls += 1
        await sleep_ms(rec["latency_ms"])
        if "error" in rec:
            self.stats.errors += 1
            raise RuntimeError(rec["error"])
        self.stats.latencies_ms.append(rec["latency_ms"])
        return cls(**rec["result"])

    def _save(self) -> None:
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps(self.cache))

    async def analyze_turn(self, transcript: str, agent_last: str = "") -> TurnAnalysis:
        return await self._call("analyze_turn", TurnAnalysis, transcript, agent_last)

    async def classify_overlap(self, agent_said: str, user_text: str) -> OverlapAnalysis:
        return await self._call("classify_overlap", OverlapAnalysis, agent_said, user_text)
