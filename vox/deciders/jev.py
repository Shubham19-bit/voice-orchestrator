"""Jev (TypeSafe AI "System One") decider.

API (docs.typesafe.ai/api):
    POST https://api.typesafe.ai/v1/systemone
    {"model": "jev-latest", "state": {...}, "questions": {id: {type, instructions, criteria}}}
    -> {"answers": {id: {"type": "noul", "noul": p} | {"type": "choice", "choice", "probabilities", "confidence"}
                     | {"type": "score", "score", "legend", "probabilities", "confidence"}}, "usage": {...}}

Key design choice: ONE request answers all three turn questions (end-of-turn, tool,
urgency) in parallel — "speculative fan-out". So when the end-of-turn answer says
"done", the tool is already known and the backend call can start immediately.

Tips baked in (from the docs' "jaggedness" notes):
  * keep one fixed phrasing per decision (answers shift with wording),
  * pin a model version (JEV_MODEL=jev-1.13.0) once thresholds are tuned,
  * Noul has no confidence field — threshold the probability directly,
  * PHI is redacted before it leaves our process (minimum-necessary principle).
"""
from __future__ import annotations

import os

from ..clock import now_ms
from ..http import post_json
from ..phi import Redactor
from ..tools import TOOLS
from .base import Decider, OverlapAnalysis, TurnAnalysis

JEV_URL = "https://api.typesafe.ai/v1/systemone"

EOT_Q = {
    "type": "noul",
    "instructions": (
        "`transcript` is a live speech-to-text transcript of a patient talking on the phone to a clinic's "
        "voice assistant. The patient has just gone quiet. Has the patient finished their turn, so the "
        "assistant should reply now? Answer no if the sentence is cut off, ends with a filler word "
        "(um, uh), a conjunction or preposition, or the patient is clearly in the middle of a thought."
    ),
}
TOOL_Q = {
    "type": "choice",
    "instructions": "Which backend action does the patient's request in `transcript` need?",
    "criteria": dict(TOOLS),
}
URGENCY_Q = {
    "type": "score",
    "instructions": "How medically urgent is what the patient says in `transcript`?",
    "criteria": [
        "Routine: no symptoms, admin or information request.",
        "Needs follow-up within days: mild symptoms or side effects.",
        "Urgent: severe symptoms such as chest pain, trouble breathing, fainting, or allergic swelling.",
    ],
}
OVERLAP_Q = {
    "type": "choice",
    "instructions": (
        "The assistant was speaking (`assistant_said`) when the patient spoke over it (`patient_said`). "
        "What did the patient do?"
    ),
    "criteria": {
        "backchannel": "Short acknowledgement while listening (mm-hmm, okay, yeah, right). The assistant should keep talking.",
        "interruption": "The patient wants to say something, ask, correct or stop the assistant. The assistant should stop and listen.",
        "other": "Noise, coughing, or talking to someone else in the room.",
    },
}


def build_turn_request(transcript: str, agent_last: str, model: str) -> dict:
    return {
        "model": model,
        "state": {"transcript": transcript, "assistant_last_said": agent_last},
        "questions": {"eot": EOT_Q, "tool": TOOL_Q, "urgency": URGENCY_Q},
    }


def parse_turn_response(data: dict) -> tuple[float, str, float, float]:
    a = data["answers"]
    eot = float(a["eot"]["noul"])
    tool = a["tool"]["choice"]
    tool_conf = float(a["tool"].get("confidence", 0.0))
    urg = float(a["urgency"]["score"])
    # score levels may be indexed 1..n or 0..n-1 depending on legend; normalise to 0..2
    legend = a["urgency"].get("legend") or {}
    try:
        lo = min(float(k) for k in legend.keys())
        urg -= lo
    except ValueError:
        pass
    return eot, tool, tool_conf, urg


def build_overlap_request(agent_said: str, user_text: str, model: str) -> dict:
    return {"model": model,
            "state": {"assistant_said": agent_said, "patient_said": user_text},
            "questions": {"overlap": OVERLAP_Q}}


def parse_overlap_response(data: dict) -> tuple[str, float]:
    a = data["answers"]["overlap"]
    return a["choice"], float(a.get("confidence", 0.0))


class JevDecider(Decider):
    name = "jev"

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 redactor: Redactor | None = None, timeout_s: float = 3.0, url: str = JEV_URL,
                 post=post_json):
        super().__init__()
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY not set — get early access at typesafe.ai, or use --decider heuristic")
        self.model = model or os.environ.get("JEV_MODEL", "jev-latest")
        self.redactor = redactor or Redactor()
        self.timeout_s = timeout_s
        self.url = url
        self._post = post

    async def _call(self, body: dict) -> dict:
        self.stats.calls += 1
        try:
            return await self._post(self.url, body, {"Authorization": f"Bearer {self.api_key}"}, self.timeout_s)
        except Exception:
            self.stats.errors += 1
            raise

    async def analyze_turn(self, transcript: str, agent_last: str = "") -> TurnAnalysis:
        t0 = now_ms()
        body = build_turn_request(self.redactor.redact(transcript), self.redactor.redact(agent_last), self.model)
        eot, tool, conf, urg = parse_turn_response(await self._call(body))
        lat = now_ms() - t0
        self.stats.latencies_ms.append(lat)
        return TurnAnalysis(eot, tool if tool in TOOLS else "none", conf, urg, lat, self.name)

    async def classify_overlap(self, agent_said: str, user_text: str) -> OverlapAnalysis:
        t0 = now_ms()
        body = build_overlap_request(self.redactor.redact(agent_said), self.redactor.redact(user_text), self.model)
        kind, conf = parse_overlap_response(await self._call(body))
        lat = now_ms() - t0
        self.stats.latencies_ms.append(lat)
        return OverlapAnalysis(kind, conf, lat, self.name)
