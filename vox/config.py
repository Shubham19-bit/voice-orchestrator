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
    speculate_prob: float = 0.5       # semantic: P(eot) >= this -> start pipeline, hold audio at the gate
    hold_prob: float = 0.5            # semantic: P(eot) >= this -> release audio after hold_ms, else after max_wait_ms
    hold_ms: float = 600              # ... release audio after this much silence
    scaled_wait: bool = False         # True: below hold_prob, wait hold_ms + (1-P)*(max_wait_ms-hold_ms) — a smooth
                                      #       "the less sure, the longer we wait" instead of one big step
    max_wait_ms: float = 1400         # semantic: low P(eot) -> wait up to this long
    adaptive_patience: bool = False   # learn THIS caller's pause length: never reply sooner than their longest
    patience_margin_ms: float = 250   #   recent mid-sentence pause + margin (capped) — slow speakers get more time
    patience_cap_ms: float = 1500
    speculate: bool = True
    # --- tool routing -----------------------------------------------------
    routing: str = "llm"              # "llm": function-calling round trip | "decider": routed by fast decider
    # --- barge-in ---------------------------------------------------------
    bargein: str = "vad"              # "vad": any speech >= vad_bargein_ms stops agent | "semantic": classify it
    vad_bargein_ms: float = 300
    bargein_force_ms: float = 1500    # semantic: talking this long over the agent always stops it
    duck_on_overlap: bool = False     # semantic barge-in: drop agent volume the instant the patient speaks,
                                      #   then stop (interruption) or restore (backchannel) once we know which
    merge_window_ms: float = 800      # patient talks over us within this long -> we cut them off; merge into one turn
    # --- streaming --------------------------------------------------------
    first_chunk_words: int = 0        # >0: send the first TTS chunk at the first comma once this many words
                                      #     have arrived (clause-level), instead of waiting for a full sentence

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
    # v2: (a) speculate even when unsure (P>=0.2) so the pipeline runs DURING the long wait —
    #     the tail becomes max(wait, pipeline) instead of wait + pipeline;
    #     (b) clause-level TTS: start speaking at the first comma instead of the first full stop;
    #     (c) wait scales smoothly with uncertainty; (d) duck agent volume instantly on overlap.
    "hybrid-v2": Strategy("hybrid-v2", turn_mode="semantic", speculate=True, speculate_prob=0.2, hold_prob=0.5,
                          routing="decider", bargein="semantic", first_chunk_words=4, scaled_wait=True,
                          duck_on_overlap=True),
    # Tried and rejected: adaptive patience (cut-ins 4%->3% but p50 +76 ms, p90 +280 ms for every
    # caller who ever pauses). Kept as an option: Strategy(..., adaptive_patience=True).
}
