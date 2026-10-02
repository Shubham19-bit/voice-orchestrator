"""Live mode: talk to the agent with your mic. Needs DEEPGRAM_API_KEY + LLM_API_KEY
(+ TYPESAFE_API_KEY for --decider jev). Runs on a normal laptop CPU.

    mic -> energy VAD ──────────────┐
        └> Deepgram streaming STT ──┴> Orchestrator -> decider / tools / Groq LLM -> Deepgram TTS -> speaker
"""
from __future__ import annotations

import asyncio
import os
import random
import statistics
from pathlib import Path

from .clock import now_ms
from .config import PRESETS
from .deciders import make_decider
from .orchestrator import Orchestrator
from .phi import Redactor
from .tools import FakeClinicBackend, ToolRunner
from .tracing import Tracer

GREETING = "Hi, this is the care assistant from Lakeside Clinic. How can I help you today?"


async def run_live(strategy_name: str, decider_kind: str, out_dir: Path) -> None:
    missing = [k for k in ("DEEPGRAM_API_KEY", "LLM_API_KEY") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"Missing {missing} in .env — see README 'Live mode'.")

    from .providers.audio_io import EnergyVAD, Mic, SpeakerOut
    from .providers.deepgram import DeepgramSTT, DeepgramTTS
    from .providers.openai_llm import OpenAICompatLLM

    strategy = PRESETS[strategy_name]
    redactor = Redactor([os.environ["PATIENT_NAME"]] if os.environ.get("PATIENT_NAME") else [])
    tracer = Tracer(redactor, conversation_id="live")
    llm, tts, speaker = OpenAICompatLLM(), DeepgramTTS(), SpeakerOut()
    decider = make_decider(decider_kind, budget_ms=float(os.environ.get("DECIDER_BUDGET_MS", 350)))
    orch = Orchestrator(strategy, decider, llm, tts, speaker, ToolRunner(FakeClinicBackend(random.Random())),
                        tracer, log=print)

    latencies: list[float] = []

    def on_start(r):
        if r.t_silence is not None:
            lat = r.t_audio_start - r.t_silence
            latencies.append(lat)
            stages = []
            if r.t_route_end:
                stages.append(f"route {r.t_route_end - r.t_trigger:.0f}")
            if r.t_tool_start:
                stages.append(f"tool {r.t_tool_end - r.t_tool_start:.0f}")
            if r.t_llm_first:
                stages.append(f"llm-ttft {r.t_llm_first - r.t_llm_start:.0f}")
                stages.append(f"tts {r.t_ready - r.t_llm_first:.0f}")
            print(f"   ⏱  {lat:.0f} ms after you stopped  (wait {r.t_trigger - r.t_silence:.0f} | "
                  f"{' | '.join(stages)})  tool={r.tool}{' [speculative]' if r.speculative else ''}")

    orch.on_audio_start.append(on_start)

    stt = DeepgramSTT(orch.on_words)
    await stt.connect()
    vad = EnergyVAD(on_start=orch.on_voice_start, on_stop=orch.on_voice_stop)

    print(f"\nStrategy={strategy.name}  decider={decider.name}  (Ctrl+C to stop)\n🎧 Use headphones.\n")
    async for chunk in tts.synth(GREETING):
        await speaker.play(chunk)
    orch.agent_last = GREETING

    mic = Mic()
    mic.start()
    try:
        while True:
            t, frame = await mic.queue.get()
            vad.process(t, frame)
            await stt.send(frame)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        mic.stop()
        await orch.shutdown()
        await stt.close()
        tracer.export(out_dir, f"live-{int(now_ms())}")
        if latencies:
            print(f"\nResponse latency over {len(latencies)} turns: p50={statistics.median(latencies):.0f} ms, "
                  f"max={max(latencies):.0f} ms.  Decider: {decider.stats.calls} calls, "
                  f"{decider.stats.fallbacks} fallbacks.  Trace → {out_dir}")
        await llm.aclose()
        await tts.aclose()
        await decider.aclose()
        speaker.close()
