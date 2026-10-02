"""Orchestration strategies. Each is one point in the latency vs. interruption trade-off."""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Strategy:
    name: str
    # --- end-of-turn ------------------------------------------------------
    turn_mode: str = "silence"        # "silence": fixed timeout | "semantic": ask the decider
    silence_timeout_ms: float = 700   # silence mode: reply after this much silence
    check_at_ms: float = 200          # semantic: first decision after this much silence
    commit_prob: float = 0.8          # semantic: P(eot) >= this -> reply now
    speculate_prob: float = 0.5       # semantic: P(eot) >= this -> start pipeline, hold audio
    hold_ms: float = 600              # ... and release audio after this much silence
    max_wait_ms: float = 1400         # semantic: low P(eot) -> wait up to this long
    speculate: bool = True
    # --- tool routing -----------------------------------------------------
    routing: str = "llm"              # "llm": function-calling round trip | "decider": routed by fast decider
    # --- barge-in ---------------------------------------------------------
    bargein: str = "vad"              # "vad": any speech >= vad_bargein_ms stops agent | "semantic": classify it
    vad_bargein_ms: float = 300
    bargein_force_ms: float = 1500    # semantic: talking this long over the agent always stops it
    merge_window_ms: float = 800      # patient talks over us within this long -> we cut them off; merge into one turn

    def with_(self, **kw) -> "Strategy":
        return replace(self, **kw)


PRESETS: dict[str, Strategy] = {
    # What most voice-agent stacks ship with by default
    "baseline-700": Strategy("baseline-700", turn_mode="silence", silence_timeout_ms=700, routing="llm", bargein="vad"),
    # "Just lower the timeout" — faster, but cuts patients off mid-thought
    "aggressive-400": Strategy("aggressive-400", turn_mode="silence", silence_timeout_ms=400, routing="llm", bargein="vad"),
    # Ablation: semantic end-of-turn only (no speculation, LLM routing, VAD barge-in)
    "semantic-eot": Strategy("semantic-eot", turn_mode="semantic", speculate=False, routing="llm", bargein="vad"),
    # Full hybrid: semantic end-of-turn + speculative pipeline + decider routing + semantic barge-in
    "hybrid": Strategy("hybrid", turn_mode="semantic", speculate=True, routing="decider", bargein="semantic"),
}
