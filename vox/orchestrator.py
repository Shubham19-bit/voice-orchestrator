"""The orchestration layer: sequences STT -> decisions -> tools -> LLM -> TTS -> speaker.

Inputs (from the simulator or the live mic/STT adapter):
    on_voice_start(t)          VAD: patient started making sound
    on_voice_stop(t_silence)   VAD: patient went quiet at t_silence
    on_words(words)            STT: current word hypotheses (with timestamps)

Key ideas, each mapped to the 100ms project description:
  * Latency budgeting  — every stage is timestamped on the Response record + traced.
  * Turn-taking        — "silence" (fixed timeout) vs "semantic" (ask the decider
                         whether the patient is done, re-ask as transcript updates).
  * Speculation        — at medium confidence, start tool+LLM+TTS early but HOLD the
                         audio behind a gate. Patient resumes -> cancel silently (no
                         audible cut-in, just a wasted call). Confident -> open the gate.
  * Barge-in           — patient talks over the agent: "vad" stops on any speech,
                         "semantic" classifies backchannel ("mm-hmm") vs interruption.
  * Graceful recovery  — decider has a latency budget + local fallback; tools have
                         timeouts + spoken fallbacks; cancellations are clean.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Callable

from .clock import now_ms, sleep_until_ms
from .config import Strategy
from .deciders.base import Decider, TurnAnalysis
from .deciders.heuristic import overlap_kind
from .providers.base import LLM, TTS, AudioChunk, AudioOut
from .tools import ToolRunner
from .tracing import Tracer

SYSTEM_PROMPT = (
    "You are a friendly voice assistant calling patients on behalf of a clinic. Speak in short, natural "
    "sentences (this is a phone call). Use the tool result if one is given. Never give medical advice "
    "beyond the tool result; for symptoms, say a nurse will follow up."
)


@dataclass
class Word:
    text: str
    start_ms: float
    end_ms: float


@dataclass
class Response:
    id: int
    turn_text: str
    speculative: bool
    analysis: TurnAnalysis | None
    t_silence: float | None
    t_trigger: float
    gate: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    tool: str = "none"
    tool_timed_out: bool = False
    reply_text: str = ""
    playing: bool = False
    outcome: str = "pending"   # played | interrupted | interrupted_merged | cancelled_spec | cancelled_late | stale
    t_commit: float | None = None
    t_route_end: float | None = None
    t_tool_start: float | None = None
    t_tool_end: float | None = None
    t_llm_start: float | None = None
    t_llm_first: float | None = None
    t_tts_start: float | None = None
    t_ready: float | None = None
    t_audio_start: float | None = None
    t_audio_end: float | None = None

    @property
    def committed(self) -> bool:
        return self.gate.is_set()


@dataclass
class Overlap:
    t_start: float
    response_id: int
    kind: str = "?"
    stopped: bool = False
    t_stop: float | None = None
    t_quiet: float | None = None     # when the agent became quiet (ducked or stopped)
    text: str = ""


class Orchestrator:
    def __init__(self, strategy: Strategy, decider: Decider, llm: LLM, tts: TTS, audio_out: AudioOut,
                 tools: ToolRunner, tracer: Tracer | None = None, system_prompt: str = SYSTEM_PROMPT,
                 log: Callable[[str], None] | None = None):
        self.s = strategy
        self.decider, self.llm, self.tts, self.audio_out, self.tools = decider, llm, tts, audio_out, tools
        self.tracer = tracer or Tracer()
        self.system_prompt = system_prompt
        self.log = log or (lambda msg: None)

        self.words: list[Word] = []
        self.turn_reset_ms = float("-inf")
        self.pauses: list[float] = []    # this caller's observed mid-turn pauses (adaptive patience)
        self.turn_prefix = ""            # carried-over text when we cut the patient off (see _barge_in_stop)
        self.turn_has_voice = False
        self.user_speaking = False
        self.voice_start_ms: float | None = None
        self.silence_start_ms: float | None = None

        self.resp: Response | None = None
        self.responses: list[Response] = []
        self.overlaps: list[Overlap] = []
        self.history: list[dict] = []
        self.agent_last = ""

        self._silence_task: asyncio.Task | None = None
        self._overlap_task: asyncio.Task | None = None
        self._overlap: Overlap | None = None
        self._words_changed = asyncio.Event()
        self._rid = 0
        self.on_audio_start: list[Callable[[Response], None]] = []
        self.on_audio_end: list[Callable[[Response], None]] = []

    # ================================================================ inputs
    def on_words(self, words: list[Word]) -> None:
        self.words = words
        ev, self._words_changed = self._words_changed, asyncio.Event()
        ev.set()

    def on_voice_start(self, t: float | None = None) -> None:
        t = now_ms() if t is None else t
        if self.silence_start_ms is not None and self.turn_has_voice and not (self.resp and self.resp.playing):
            self._note_pause(t - self.silence_start_ms)  # they paused, then carried on: a mid-turn pause
        self.user_speaking, self.voice_start_ms, self.silence_start_ms = True, t, None
        self.tracer.event("voice_start", "user")
        self._cancel(self._silence_task)
        r = self.resp
        if r and not r.playing:
            # patient kept talking: drop the prepared reply before it becomes audible
            self._cancel_response(r, "cancelled_late" if r.committed else "cancelled_spec")
            self.turn_has_voice = True
        elif r and r.playing:
            if self._overlap_task is None or self._overlap_task.done():
                self._overlap = Overlap(t, r.id)
                self.overlaps.append(self._overlap)
                self._overlap_task = asyncio.create_task(self._handle_overlap(r, self._overlap))
        else:
            self.turn_has_voice = True

    def on_voice_stop(self, t_silence: float | None = None) -> None:
        t = now_ms() if t_silence is None else t_silence
        self.user_speaking, self.silence_start_ms = False, t
        self.tracer.event("voice_stop", "user")
        if self.resp and self.resp.playing:
            return  # overlap speech; the barge-in handler owns it
        if self.turn_has_voice:
            self._cancel(self._silence_task)
            self._silence_task = asyncio.create_task(self._silence_watch(t))

    # ================================================================ helpers
    def _note_pause(self, ms: float) -> None:
        if ms >= 300:
            self.pauses.append(ms)

    def min_wait_ms(self) -> float:
        """Adaptive patience: never answer sooner than this caller's longest recent mid-turn pause."""
        if not self.s.adaptive_patience or not self.pauses:
            return 0.0
        return min(self.s.patience_cap_ms, max(self.pauses[-5:]) + self.s.patience_margin_ms)

    def turn_text(self) -> str:
        new = " ".join(w.text for w in self.words if w.start_ms >= self.turn_reset_ms).strip()
        return f"{self.turn_prefix} {new}".strip() if self.turn_prefix else new

    def _overlap_text(self, since: float) -> str:
        return " ".join(w.text for w in self.words if w.start_ms >= since - 50).strip()

    async def _wait_words(self, deadline_ms: float) -> bool:
        ev = self._words_changed
        try:
            # >=1 ms so a deadline a float-epsilon away can't spin the loop without time advancing
            await asyncio.wait_for(ev.wait(), max(1.0, deadline_ms - now_ms()) / 1000)
            return True
        except asyncio.TimeoutError:
            return False

    @staticmethod
    def _cancel(task: asyncio.Task | None) -> None:
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()

    def _messages(self, user_text: str, tool_say: str | None = None) -> list[dict]:
        msgs = [{"role": "system", "content": self.system_prompt}, *self.history[-8:],
                {"role": "user", "content": user_text}]
        if tool_say:
            msgs.append({"role": "system", "content": f"Tool result: {tool_say}"})
        return msgs

    # ================================================================ end-of-turn
    async def _silence_watch(self, t0: float) -> None:
        s = self.s
        span = self.tracer.start("end_of_turn_wait", "turn", mode=s.turn_mode)
        try:
            if s.turn_mode == "silence":
                await sleep_until_ms(t0 + s.silence_timeout_ms)
                while not self.turn_text() and await self._wait_words(t0 + s.silence_timeout_ms + 1500):
                    pass
                if self.turn_text():
                    self._start_or_commit(None, t0)
                return

            await sleep_until_ms(t0 + s.check_at_ms)
            last_text, analysis, p = None, None, 0.0
            while True:
                text = self.turn_text()
                if text and text != last_text:
                    last_text = text
                    with self.tracer.span("decide_turn", "decider", text=text) as dsp:
                        analysis = await self.decider.analyze_turn(text, self.agent_last)
                        dsp.attrs.update(eot=round(analysis.eot_prob, 3), tool=analysis.tool,
                                         urgency=analysis.urgency, fallback=analysis.fallback)
                    p = analysis.eot_prob
                    if self.resp and not self.resp.playing and self.resp.turn_text != text:
                        self._cancel_response(self.resp, "stale")
                    if p >= s.commit_prob and now_ms() >= t0 + self.min_wait_ms():
                        self._start_or_commit(analysis, t0)
                        return
                    if p >= s.commit_prob and self.resp is None:  # confident, but this caller pauses long
                        self._start_response(analysis, t0, committed=False)
                    if s.speculate and p >= s.speculate_prob and self.resp is None:
                        self._start_response(analysis, t0, committed=False)
                if p >= s.hold_prob:
                    wait = s.hold_ms
                elif s.scaled_wait:
                    wait = s.hold_ms + (1 - p) * (s.max_wait_ms - s.hold_ms)
                else:
                    wait = s.max_wait_ms
                deadline = t0 + max(wait, self.min_wait_ms())
                if now_ms() >= deadline:
                    if text:
                        self._start_or_commit(analysis if text == last_text else None, t0)
                        return
                    deadline = now_ms() + 1500  # no transcript yet: give STT a moment
                cur = self.turn_text()
                if cur and cur != last_text:
                    continue  # words arrived while we were deciding — re-decide right away
                await self._wait_words(deadline)
                if not self.turn_text() and now_ms() >= deadline:
                    return
        finally:
            self.tracer.end(span)

    def _start_or_commit(self, analysis: TurnAnalysis | None, t0: float) -> None:
        r = self.resp
        if r and not r.playing and r.turn_text == self.turn_text():
            r.gate.set()
            r.t_commit = now_ms()
            self.tracer.event("commit", "turn", response=r.id, speculative=r.speculative)
            return
        if r and not r.playing:
            self._cancel_response(r, "stale")
        self._start_response(analysis, t0, committed=True)

    def _start_response(self, analysis: TurnAnalysis | None, t0: float, committed: bool) -> Response:
        self._rid += 1
        r = Response(self._rid, self.turn_text(), speculative=not committed, analysis=analysis,
                     t_silence=t0, t_trigger=now_ms())
        if committed:
            r.gate.set()
            r.t_commit = r.t_trigger
        self.tracer.event("response_start", "turn", response=r.id, speculative=not committed, text=r.turn_text)
        r.task = asyncio.create_task(self._run_response(r))
        self.resp = r
        self.responses.append(r)
        return r

    def _cancel_response(self, r: Response, outcome: str) -> None:
        r.outcome = outcome
        self.tracer.event(f"response_{outcome}", "turn", response=r.id)
        self._cancel(r.task)
        if self.resp is r:
            self.resp = None

    # ================================================================ response pipeline
    async def _run_response(self, r: Response) -> None:
        producer: asyncio.Task | None = None
        try:
            # 1) routing
            if self.s.routing == "decider":
                a = r.analysis
                if a is None:
                    with self.tracer.span("decide_route", "decider", text=r.turn_text):
                        a = await self.decider.analyze_turn(r.turn_text, self.agent_last)
                    r.analysis = a
                r.tool = a.tool
            else:
                with self.tracer.span("llm_tool_call", "llm") as sp:
                    r.tool = await self.llm.choose_tool(self._messages(r.turn_text))
                    sp.attrs["tool"] = r.tool
            r.t_route_end = now_ms()

            # 2) tool call (with timeout + spoken fallback)
            tool_say = None
            if r.tool and r.tool != "none":
                r.t_tool_start = now_ms()
                with self.tracer.span(f"tool:{r.tool}", "tool") as sp:
                    res = await self.tools.run(r.tool, r.turn_text)
                    if res:
                        sp.attrs.update(ok=res.ok, timed_out=res.timed_out)
                        r.tool_timed_out, tool_say = res.timed_out, res.say
                r.t_tool_end = now_ms()

            # 3) LLM stream -> sentence splitter -> TTS -> queue
            queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue()
            producer = asyncio.create_task(self._produce_audio(r, self._messages(r.turn_text, tool_say), queue))
            first = await queue.get()
            r.t_ready = now_ms()

            # 4) speculation gate: audio is ready but we only speak once committed
            if not r.gate.is_set():
                with self.tracer.span("held_at_gate", "turn", response=r.id):
                    await r.gate.wait()
            if first is None:
                r.outcome = "empty"
                return

            # 5) playback
            self._playback_started(r)
            chunk = first
            with self.tracer.span("playback", "playback", response=r.id):
                while chunk is not None:
                    await self.audio_out.play(chunk)
                    chunk = await queue.get()
            r.outcome = "played"
        finally:
            if producer:
                producer.cancel()
            if r.playing:
                self._playback_ended(r)
            if self.resp is r:
                self.resp = None

    async def _produce_audio(self, r: Response, msgs: list[dict], queue: asyncio.Queue) -> None:
        buf = ""
        r.t_llm_start = now_ms()
        llm_span = self.tracer.start("llm_ttft", "llm")
        async for piece in self.llm.stream(msgs):
            if r.t_llm_first is None:
                r.t_llm_first = now_ms()
                self.tracer.end(llm_span)
            buf += piece
            r.reply_text += piece
            # flush on sentence end (or a long clause) so TTS starts as early as possible
            while True:
                first = r.t_tts_start is None and self.s.first_chunk_words > 0
                clause_words = self.s.first_chunk_words if first else 8
                m = (re.search(r"[.!?](\s|$)", buf)
                     or (re.search(r"[,;:]\s", buf) if len(buf.split()) >= clause_words else None))
                if not m:
                    break
                sent, buf = buf[:m.end()], buf[m.end():]
                await self._synth(r, sent, queue)
        if r.t_llm_first is None:
            self.tracer.end(llm_span)
        if buf.strip():
            await self._synth(r, buf, queue)
        await queue.put(None)

    async def _synth(self, r: Response, sentence: str, queue: asyncio.Queue) -> None:
        first_sentence = r.t_tts_start is None
        if first_sentence:
            r.t_tts_start = now_ms()
            sp = self.tracer.start("tts_ttfb", "tts")
        got_first = False
        async for chunk in self.tts.synth(sentence.strip()):
            if first_sentence and not got_first:
                self.tracer.end(sp)
                got_first = True
            await queue.put(chunk)

    def _playback_started(self, r: Response) -> None:
        r.playing, r.t_audio_start = True, now_ms()
        self.turn_reset_ms = r.t_audio_start
        self.turn_prefix = ""
        self.turn_has_voice = False
        self._cancel(self._silence_task)
        self.history.append({"role": "user", "content": r.turn_text})
        self.log(f"[agent ▶] responding to: {r.turn_text!r}")
        for cb in self.on_audio_start:
            cb(r)

    def _playback_ended(self, r: Response) -> None:
        r.playing, r.t_audio_end = False, now_ms()
        interrupted = r.outcome.startswith("interrupted")
        if r.outcome != "interrupted_merged":   # a merged cut-in never "happened" conversationally
            self.agent_last = r.reply_text.strip()
            self.history.append({"role": "assistant",
                                 "content": self.agent_last + (" (interrupted)" if interrupted else "")})
        if not interrupted:
            ov = self._overlap
            if self._overlap_task and not self._overlap_task.done() and ov:
                self._cancel(self._overlap_task)
                text = self._overlap_text(ov.t_start)
                if self.user_speaking or (text and overlap_kind(text)[0] == "interruption"):
                    # they were starting their next turn as we finished
                    self.turn_reset_ms = ov.t_start - 50
                    self.turn_has_voice = True
                    if not self.user_speaking and self.silence_start_ms is not None:
                        self._silence_task = asyncio.create_task(self._silence_watch(self.silence_start_ms))
                else:
                    self.turn_reset_ms = now_ms()
            else:
                self.turn_reset_ms = now_ms()
        for cb in self.on_audio_end:
            cb(r)

    # ================================================================ barge-in
    async def _handle_overlap(self, r: Response, ov: Overlap) -> None:
        s = self.s
        span = self.tracer.start("overlap", "bargein", mode=s.bargein)
        try:
            if s.bargein == "vad":
                await asyncio.sleep(s.vad_bargein_ms / 1000)
                if self.user_speaking and self.voice_start_ms == ov.t_start and r.playing:
                    ov.kind = "speech>=%dms" % s.vad_bargein_ms
                    self._barge_in_stop(r, ov)
                else:
                    ov.kind = "short-noise"
                return

            if s.duck_on_overlap and r.playing:
                self.audio_out.duck(True)
                ov.t_quiet = now_ms()
                self.tracer.event("duck", "bargein", response=r.id)
            last = None
            while r.playing:
                text = self._overlap_text(ov.t_start)
                if text and text != last:
                    last = text
                    with self.tracer.span("decide_overlap", "decider", text=text) as dsp:
                        a = await self.decider.classify_overlap(r.reply_text, text)
                        dsp.attrs.update(kind=a.kind, fallback=a.fallback)
                    ov.kind, ov.text = a.kind, text
                    if a.kind == "interruption" and r.playing:
                        self._barge_in_stop(r, ov)
                        return
                if self.user_speaking and now_ms() - ov.t_start >= s.bargein_force_ms and r.playing:
                    ov.kind = "long-speech"
                    self._barge_in_stop(r, ov)
                    return
                if (not self.user_speaking and last is not None and self.silence_start_ms is not None
                        and now_ms() - self.silence_start_ms > 250):
                    # it was a backchannel / noise: keep talking, forget those words
                    self.turn_reset_ms = now_ms()
                    if s.duck_on_overlap:
                        self.audio_out.duck(False)
                    return
                await self._wait_words(now_ms() + 100)
        finally:
            self.tracer.end(span, kind=ov.kind, stopped=ov.stopped)

    def _barge_in_stop(self, r: Response, ov: Overlap) -> None:
        ov.stopped, ov.t_stop = True, now_ms()
        ov.t_quiet = ov.t_quiet or ov.t_stop
        self.audio_out.duck(False)  # next reply starts at full volume
        self.tracer.event("barge_in_stop", "bargein", response=r.id, after_ms=round(ov.t_stop - ov.t_start))
        self.log(f"[agent ■] stopped by barge-in after {ov.t_stop - ov.t_start:.0f} ms")
        self.audio_out.stop()
        r.outcome = "interrupted"
        self._cancel(r.task)
        # the overlapping words start the patient's new turn
        self.turn_reset_ms = ov.t_start - 50
        if ov.t_start - (r.t_audio_start or 0) < self.s.merge_window_ms:
            # we most likely cut them off mid-thought: treat it as ONE turn, so the
            # decider/LLM see "I'm not sure when my next appointment is", not two fragments
            self.turn_prefix = r.turn_text
            if r.t_silence is not None:
                self._note_pause(ov.t_start - r.t_silence)  # we cut them off: that pause was mid-turn
            if self.history[-1:] == [{"role": "user", "content": r.turn_text}]:
                self.history.pop()
            r.outcome = "interrupted_merged"
        self.turn_has_voice = True
        if not self.user_speaking and self.silence_start_ms is not None:
            self._silence_task = asyncio.create_task(self._silence_watch(self.silence_start_ms))

    async def shutdown(self) -> None:
        for t in [self._silence_task, self._overlap_task, *(r.task for r in self.responses)]:
            self._cancel(t)
        await asyncio.sleep(0)
