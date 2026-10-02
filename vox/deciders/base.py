"""The decision layer: fast, typed decisions the orchestrator needs on the hot path.

    analyze_turn(transcript)        -> is the patient done? which tool? how urgent?
    classify_overlap(agent, user)   -> was that a backchannel ("mm-hmm") or a real interruption?

Generation (what to say) stays with the LLM. Decisions go here, because a decision
model like Jev answers in ~100-300 ms instead of seconds, and answers are typed
(no parsing, no hallucinated tool names).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TurnAnalysis:
    eot_prob: float                 # P(patient finished their turn)
    tool: str = "none"              # one of tools.TOOLS
    tool_conf: float = 0.0
    urgency: float = 0.0            # 0 routine .. 2 urgent
    latency_ms: float = 0.0
    source: str = ""
    fallback: bool = False


@dataclass
class OverlapAnalysis:
    kind: str                       # "backchannel" | "interruption" | "other"
    conf: float = 0.0
    latency_ms: float = 0.0
    source: str = ""
    fallback: bool = False


@dataclass
class DeciderStats:
    latencies_ms: list[float] = field(default_factory=list)
    calls: int = 0
    errors: int = 0
    fallbacks: int = 0


class Decider:
    name = "base"

    def __init__(self) -> None:
        self.stats = DeciderStats()

    async def analyze_turn(self, transcript: str, agent_last: str = "") -> TurnAnalysis:
        raise NotImplementedError

    async def classify_overlap(self, agent_said: str, user_text: str) -> OverlapAnalysis:
        raise NotImplementedError

    async def aclose(self) -> None:
        pass
