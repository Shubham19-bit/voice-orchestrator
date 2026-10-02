"""Graceful degradation: a hard latency budget on the remote decider.

If Jev (or any primary) doesn't answer within `budget_ms`, or errors (429/529/network),
we answer from the local heuristic instead and carry on. The patient never waits on
a slow decision. A simple circuit breaker skips the primary for a cooldown after
repeated failures, so a dead dependency doesn't add `budget_ms` to every turn.
"""
from __future__ import annotations

import asyncio

from ..clock import now_ms
from .base import Decider, OverlapAnalysis, TurnAnalysis


class FallbackDecider(Decider):
    def __init__(self, primary: Decider, secondary: Decider, budget_ms: float = 350,
                 trip_after: int = 3, cooldown_ms: float = 10_000):
        super().__init__()
        self.primary, self.secondary = primary, secondary
        self.name = f"{primary.name}+fallback"
        self.budget_ms = budget_ms
        self.trip_after, self.cooldown_ms = trip_after, cooldown_ms
        self._consecutive_failures = 0
        self._open_until = -1.0

    def _breaker_open(self) -> bool:
        return now_ms() < self._open_until

    def _failed(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.trip_after:
            self._open_until = now_ms() + self.cooldown_ms
            self._consecutive_failures = 0

    async def _run(self, method: str, *args):
        self.stats.calls += 1
        t0 = now_ms()
        if not self._breaker_open():
            try:
                res = await asyncio.wait_for(getattr(self.primary, method)(*args), self.budget_ms / 1000)
                self._consecutive_failures = 0
                self.stats.latencies_ms.append(now_ms() - t0)
                return res
            except (asyncio.TimeoutError, Exception):
                self.stats.errors += 1
                self._failed()
        res = await getattr(self.secondary, method)(*args)
        res.fallback = True
        res.latency_ms = now_ms() - t0
        self.stats.fallbacks += 1
        self.stats.latencies_ms.append(res.latency_ms)
        return res

    async def analyze_turn(self, transcript: str, agent_last: str = "") -> TurnAnalysis:
        return await self._run("analyze_turn", transcript, agent_last)

    async def classify_overlap(self, agent_said: str, user_text: str) -> OverlapAnalysis:
        return await self._run("classify_overlap", agent_said, user_text)

    async def aclose(self) -> None:
        await self.primary.aclose()
        await self.secondary.aclose()
