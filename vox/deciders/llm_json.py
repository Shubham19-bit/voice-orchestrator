"""Decider backed by a general chat LLM in JSON mode (OpenAI-compatible API, e.g. Groq).

This is the "what most teams do today" comparison point for Jev: same questions,
answered by a generative model that has to emit and then be parsed as JSON.
"""
from __future__ import annotations

import json
import os

from ..clock import now_ms
from ..http import post_json
from ..phi import Redactor
from ..tools import TOOLS
from .base import Decider, OverlapAnalysis, TurnAnalysis

TURN_PROMPT = """You are the turn-taking and routing module of a clinic voice assistant.
The patient's live transcript so far is below; they have just gone quiet.
Return ONLY a JSON object with keys:
  "eot_prob": probability 0-1 that the patient has finished their turn (low if cut off, ends in um/uh/and/but, or mid-thought),
  "tool": one of %s,
  "urgency": 0 (routine), 1 (mild symptoms, follow up in days) or 2 (urgent: chest pain, can't breathe, fainting, swelling).
Tool meanings: %s"""

OVERLAP_PROMPT = """The voice assistant was speaking when the patient spoke over it.
Return ONLY a JSON object {"kind": "backchannel"|"interruption"|"other"}.
backchannel = short acknowledgement (mm-hmm, okay, yeah); interruption = they want to talk/ask/correct; other = noise."""


class LLMJsonDecider(Decider):
    name = "llm-json"

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None,
                 redactor: Redactor | None = None, timeout_s: float = 5.0, post=post_json):
        super().__init__()
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY not set")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")).rstrip("/")
        self.model = model or os.environ.get("DECIDER_LLM_MODEL") or os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
        self.redactor = redactor or Redactor()
        self.timeout_s = timeout_s
        self._post = post

    async def _json(self, system: str, user: str) -> dict:
        self.stats.calls += 1
        body = {"model": self.model, "temperature": 0, "max_tokens": 80,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        try:
            data = await self._post(f"{self.base_url}/chat/completions", body,
                                    {"Authorization": f"Bearer {self.api_key}"}, self.timeout_s)
            return json.loads(data["choices"][0]["message"]["content"])
        except Exception:
            self.stats.errors += 1
            raise

    async def analyze_turn(self, transcript: str, agent_last: str = "") -> TurnAnalysis:
        t0 = now_ms()
        sys = TURN_PROMPT % (list(TOOLS), json.dumps(TOOLS))
        out = await self._json(sys, f"Assistant last said: {self.redactor.redact(agent_last)!r}\n"
                                    f"Patient transcript: {self.redactor.redact(transcript)!r}")
        lat = now_ms() - t0
        self.stats.latencies_ms.append(lat)
        tool = out.get("tool", "none")
        return TurnAnalysis(float(out.get("eot_prob", 0.5)), tool if tool in TOOLS else "none", 0.0,
                            float(out.get("urgency", 0)), lat, self.name)

    async def classify_overlap(self, agent_said: str, user_text: str) -> OverlapAnalysis:
        t0 = now_ms()
        out = await self._json(OVERLAP_PROMPT, f"Assistant was saying: {self.redactor.redact(agent_said)!r}\n"
                                               f"Patient said: {self.redactor.redact(user_text)!r}")
        lat = now_ms() - t0
        self.stats.latencies_ms.append(lat)
        kind = out.get("kind", "other")
        return OverlapAnalysis(kind if kind in ("backchannel", "interruption", "other") else "other", 0.0, lat, self.name)
