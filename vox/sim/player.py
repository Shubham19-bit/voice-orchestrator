"""Plays a scripted patient into the orchestrator, like a real mic + VAD + streaming STT would.

Realism knobs:
  * word durations scale with word length, plus jitter
  * VAD reports silence only after `vad_ms` (short gaps between words are ignored)
  * STT delivers each word `stt_lag` ms after it was spoken (streaming ASR lag) —
    so at the first end-of-turn check the last word may not be in the transcript yet
  * the patient reacts to the agent: waits for it to finish, or barges in on cue
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

from ..clock import now_ms, sleep_ms, sleep_until_ms
from ..orchestrator import Orchestrator, Response, Word
from .scenarios import Scenario, parse_turn


@dataclass
class TruthTurn:
    idx: int
    text: str
    start: float
    end: float
    tool: str
    urgency: int
    is_barge_in: bool = False       # scripted as an interruption
    overlapped_agent: bool = False  # agent was actually speaking when it started


@dataclass
class TruthBackchannel:
    turn_idx: int
    text: str
    start: float
    end: float


class Player:
    def __init__(self, orch: Orchestrator, scenario: Scenario, rng: random.Random,
                 stt_lag_ms: tuple[float, float] = (90, 260), vad_ms: float = 150, onset_ms: float = 40,
                 pause_jitter: float = 0.15):
        self.o, self.sc, self.rng = orch, scenario, rng
        self.stt_lag, self.vad_ms, self.onset_ms, self.pause_jitter = stt_lag_ms, vad_ms, onset_ms, pause_jitter
        self.turns: list[TruthTurn] = []
        self.backchannels: list[TruthBackchannel] = []
        self.delivered: list[Word] = []
        self._last_delivery = 0.0
        self.current_tool = "none"
        self.plays: list[Response] = []
        self.agent_playing = False
        self._changed = asyncio.Event()
        orch.on_audio_start.append(self._on_start)
        orch.on_audio_end.append(self._on_end)

    # -------------------------------------------------- agent observation
    def _bump(self) -> None:
        ev, self._changed = self._changed, asyncio.Event()
        ev.set()

    def _on_start(self, r: Response) -> None:
        self.agent_playing = True
        self.plays.append(r)
        self._bump()

    def _on_end(self, r: Response) -> None:
        self.agent_playing = False
        self._bump()

    async def _wait_for(self, pred, timeout_ms: float) -> bool:
        deadline = now_ms() + timeout_ms
        while not pred():
            if now_ms() >= deadline:
                return False
            try:
                await asyncio.wait_for(self._changed.wait(), max(1.0, deadline - now_ms()) / 1000)
            except asyncio.TimeoutError:
                return False
        return True

    def _reply_after(self, t: float) -> Response | None:
        for r in self.plays:
            if r.t_audio_start is not None and r.t_audio_start >= t:
                return r
        return None

    # -------------------------------------------------- speaking
    def _word_ms(self, w: str) -> float:
        return (130 + 45 * len(w)) * self.rng.uniform(0.85, 1.2)

    async def _speak(self, text: str) -> tuple[float, float, str]:
        loop = asyncio.get_running_loop()
        tokens = parse_turn(text)
        t_first = now_ms()
        prev_gap = None
        for j, (w, pause) in enumerate(tokens):
            ws = now_ms()
            if j == 0 or (prev_gap is not None and prev_gap >= self.vad_ms):
                loop.call_later(self.onset_ms / 1000, self.o.on_voice_start)
            await sleep_ms(self._word_ms(w))
            we = now_ms()
            deliver_at = max(self._last_delivery + 1, we + self.rng.uniform(*self.stt_lag))
            self._last_delivery = deliver_at
            loop.call_later((deliver_at - we) / 1000, self._deliver, Word(w, ws, we))
            last = j == len(tokens) - 1
            if last:
                gap = None
            elif pause is not None:
                gap = pause * self.rng.uniform(1 - self.pause_jitter, 1 + self.pause_jitter)
            else:
                gap = self.rng.uniform(40, 110)
            if gap is None or gap >= self.vad_ms:
                loop.call_later(self.vad_ms / 1000, self.o.on_voice_stop, we)
            if gap is not None:
                await sleep_ms(gap)
            prev_gap = gap
        return t_first, now_ms(), " ".join(w for w, _ in tokens)

    def _deliver(self, w: Word) -> None:
        self.delivered.append(w)
        self.o.on_words(list(self.delivered))

    async def _backchannel(self, idx: int, turn_end: float, at_ms: float, text: str) -> None:
        if not await self._wait_for(lambda: self._reply_after(turn_end) is not None, 8000):
            return
        r = self._reply_after(turn_end)
        await sleep_until_ms(r.t_audio_start + at_ms)
        if not self.agent_playing:
            return
        s, e, _ = await self._speak(text)
        self.backchannels.append(TruthBackchannel(idx, text, s, e))

    # -------------------------------------------------- main
    async def run(self) -> None:
        await sleep_ms(400)
        bg: list[asyncio.Task] = []
        for i, spec in enumerate(self.sc.turns):
            barged = False
            if i > 0:
                prev = self.turns[-1]
                if spec.barge_in_at_ms is not None:
                    if await self._wait_for(lambda: self._reply_after(prev.end) is not None, 8000):
                        r = self._reply_after(prev.end)
                        await sleep_until_ms(r.t_audio_start + spec.barge_in_at_ms)
                        barged = True
                else:
                    def done() -> bool:
                        r = self._reply_after(prev.end)
                        return r is not None and r.t_audio_end is not None and not self.agent_playing
                    await self._wait_for(done, 9000)
                    # wait for any lingering backchannel too
                    await self._wait_for(lambda: not self.agent_playing, 9000)
                    await sleep_ms(self.rng.uniform(350, 800))
            self.current_tool = spec.tool
            overlapped = self.agent_playing
            s, e, clean = await self._speak(spec.text)
            self.turns.append(TruthTurn(i, clean, s, e, spec.tool, spec.urgency, barged, overlapped))
            if spec.backchannel_at_ms is not None:
                bg.append(asyncio.create_task(self._backchannel(i, e, spec.backchannel_at_ms, spec.backchannel_text)))
        last = self.turns[-1]
        await self._wait_for(lambda: (r := self._reply_after(last.end)) is not None and r.t_audio_end is not None, 9000)
        await sleep_ms(300)
        for t in bg:
            t.cancel()
        await self.o.shutdown()
