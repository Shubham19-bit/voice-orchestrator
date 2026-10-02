"""Decider backed by mini-Jev (vox/minijev): local, free, ~1-5 ms on CPU.

One call answers end-of-turn + tool + urgency together — the same fan-out as the Jev
client, with the same answer shapes. If no trained model exists yet, it trains one
(takes a few seconds) and saves it to models/minijev.pkl.
"""
from __future__ import annotations

import time
from pathlib import Path

from ..clock import now_ms, sleep_ms
from .base import Decider, OverlapAnalysis, TurnAnalysis


class MiniJevDecider(Decider):
    name = "minijev"

    def __init__(self, path: Path | None = None, sim_compute_ms: float | None = None):
        """sim_compute_ms: in the virtual-time simulator CPU work takes zero virtual time, so we
        charge a fixed cost per call instead (default 5 ms ≈ p95 on a laptop; fixed so benchmark
        results stay deterministic). Leave None in live mode, where real time is real."""
        super().__init__()
        from ..minijev.model import DEFAULT_PATH, MiniJev
        path = path or DEFAULT_PATH
        if not path.exists():
            from ..minijev.train import main as train
            print("[minijev] no trained model found — training one now (a few seconds)…")
            train()
        self.m = MiniJev.load(path)
        self.sim_compute_ms = sim_compute_ms

    async def _charge(self, t0: float) -> None:
        if self.sim_compute_ms is not None:
            await sleep_ms(self.sim_compute_ms)

    async def analyze_turn(self, transcript: str, agent_last: str = "") -> TurnAnalysis:
        t0, v0 = time.perf_counter(), now_ms()
        eot = self.m.noul_eot(transcript)["noul"]
        tool = self.m.choice_tool(transcript)
        urg = self.m.score_urgency(transcript)["score"]
        await self._charge(t0)
        lat = max(now_ms() - v0, (time.perf_counter() - t0) * 1000)
        self.stats.calls += 1
        self.stats.latencies_ms.append(lat)
        return TurnAnalysis(eot, tool["choice"], tool["confidence"], urg, lat, self.name)

    async def classify_overlap(self, agent_said: str, user_text: str) -> OverlapAnalysis:
        t0, v0 = time.perf_counter(), now_ms()
        a = self.m.choice_overlap(user_text)
        await self._charge(t0)
        lat = max(now_ms() - v0, (time.perf_counter() - t0) * 1000)
        self.stats.calls += 1
        self.stats.latencies_ms.append(lat)
        return OverlapAnalysis(a["choice"], a["confidence"], lat, self.name)
