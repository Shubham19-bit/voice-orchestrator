"""Rule-based decider. Runs offline, costs nothing — used as

  1. the fallback when the remote decider is slow or down (graceful degradation), and
  2. a stand-in for Jev while you wait for API access.

`simulated_latency_ms` lets you pretend it is a remote call (for fair benchmarks).
"""
from __future__ import annotations

import random
import re

from ..clock import now_ms, sleep_ms
from .base import Decider, OverlapAnalysis, TurnAnalysis

INCOMPLETE_ENDINGS = {
    "um", "uh", "er", "erm", "hmm", "like", "and", "but", "or", "so", "because", "cause", "the", "a", "an",
    "my", "your", "to", "of", "for", "with", "in", "on", "at", "about", "that", "which", "if", "when",
    "is", "was", "i", "it's", "i'm", "i've", "then", "also", "maybe", "should", "could", "can", "will",
    "would", "do", "did", "have", "had", "from", "after", "before", "um,", "uh,",
}
SHORT_COMPLETE = {"yes", "yeah", "yep", "no", "nope", "okay", "ok", "sure", "thanks", "thank you",
                  "that's right", "correct", "right", "bye", "goodbye", "hello", "hi"}
QUESTION_STARTS = ("when", "what", "where", "which", "how", "who", "why", "can", "could", "do", "does",
                   "is", "are", "should", "will", "would")
QUESTION_RE = re.compile(r"\b(when|what|where|which|how|who|why)\b|\b(can|could|do|does|did|should|will|would|is|are) "
                         r"(you|i|we|it|that|there|this)\b")
SOFT_ENDINGS = {"that", "it", "is", "i", "do", "it's"}   # fine at the end of a question ("which day was that")
BACKCHANNELS = {"mm-hmm", "mhm", "uh-huh", "okay", "ok", "yeah", "yes", "right", "sure", "got it", "i see",
                "mm", "hmm", "alright", "uh huh", "yep"}

TOOL_KEYWORDS = [
    ("report_side_effect", r"\b(dizz|nause|sick|vomit|rash|pain|hurt|headache|swell|breath|faint|side effect|feel(?:ing)? (?:weird|bad|off|unwell|strange))"),
    ("reschedule_appointment", r"\b(reschedul|move|change|cancel|push|different (?:day|time)|book)"),
    ("lookup_appointment", r"\b(appointment|visit|when do i|what time|which clinic|next checkup)"),
    ("medication_info", r"\b(medic|pill|tablet|dose|dosage|refill|pharmac|metformin|lisinopril|statin|insulin|missed|take (?:it|two|one|them))"),
]
URGENT = r"\b(chest pain|can't breathe|cannot breathe|trouble breathing|short(?:ness)? of breath|faint|passed out|swelling|throat|severe|blood)"
SOON = r"\b(dizz|nause|vomit|rash|headache|pain|side effect)"


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z'\-]*", text.lower())


def eot_probability(text: str) -> float:
    w = _words(text)
    if not w:
        return 0.0
    if text.rstrip().endswith("?"):
        return 0.92
    if w[-1] in INCOMPLETE_ENDINGS and not (w[-1] in SOFT_ENDINGS and QUESTION_RE.search(" ".join(w))):
        return 0.06
    joined = " ".join(w)
    if len(w) <= 4 and any(joined == p or joined.endswith(" " + p) for p in SHORT_COMPLETE):
        return 0.9
    if (w[0] in QUESTION_STARTS and len(w) >= 3) or (len(w) >= 3 and QUESTION_RE.search(joined)):
        return 0.85
    if len(w) <= 2:
        return 0.85 if w[0] in SHORT_COMPLETE else 0.35
    return 0.65


def route_tool(text: str) -> tuple[str, float]:
    t = text.lower()
    for tool, pat in TOOL_KEYWORDS:
        if re.search(pat, t):
            return tool, 0.7
    return "none", 0.5


def urgency(text: str) -> float:
    t = text.lower()
    if re.search(URGENT, t):
        return 2.0
    if re.search(SOON, t):
        return 1.0
    return 0.0


def overlap_kind(user_text: str) -> tuple[str, float]:
    w = _words(user_text)
    if not w:
        return "other", 0.3
    joined = " ".join(w)
    if joined in BACKCHANNELS or (len(w) <= 2 and all(x in BACKCHANNELS for x in w)):
        return "backchannel", 0.85
    if w[0] in {"wait", "sorry", "hold", "no", "actually", "but", "excuse", "stop", "hang"} or len(w) >= 2:
        return "interruption", 0.8
    return "other", 0.4


class HeuristicDecider(Decider):
    name = "heuristic"

    def __init__(self, simulated_latency_ms: float = 0.0, rng: random.Random | None = None):
        super().__init__()
        self.sim_lat = simulated_latency_ms
        self.rng = rng or random.Random(1)

    async def _fake_network(self) -> None:
        if self.sim_lat > 0:
            await sleep_ms(self.sim_lat * self.rng.lognormvariate(0, 0.35))

    async def analyze_turn(self, transcript: str, agent_last: str = "") -> TurnAnalysis:
        t0 = now_ms()
        await self._fake_network()
        tool, conf = route_tool(transcript)
        res = TurnAnalysis(eot_probability(transcript), tool, conf, urgency(transcript),
                           now_ms() - t0, self.name)
        self.stats.calls += 1
        self.stats.latencies_ms.append(res.latency_ms)
        return res

    async def classify_overlap(self, agent_said: str, user_text: str) -> OverlapAnalysis:
        t0 = now_ms()
        await self._fake_network()
        kind, conf = overlap_kind(user_text)
        res = OverlapAnalysis(kind, conf, now_ms() - t0, self.name)
        self.stats.calls += 1
        self.stats.latencies_ms.append(res.latency_ms)
        return res
