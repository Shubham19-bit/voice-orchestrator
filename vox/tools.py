"""Backend tools the agent can call, with explicit contracts.

Each tool has: a description (used by deciders/LLM for routing), a timeout, and a
graceful fallback answer if the backend is slow or down — so a degraded backend
never leaves the patient listening to dead air.

The backend here is a fake clinic system with realistic latency. Swap
`FakeClinicBackend` for real HTTP/gRPC calls later; the contract stays the same.
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

from .clock import now_ms, sleep_ms

TOOLS: dict[str, str] = {
    "lookup_appointment": "Patient asks when or where their next appointment is, or wants to confirm it.",
    "reschedule_appointment": "Patient wants to move, change, book or cancel an appointment.",
    "medication_info": "Questions about their medication: dose, timing, refills, missed doses, pharmacy.",
    "report_side_effect": "Patient describes a symptom, side effect or feeling unwell after their medication.",
    "none": "Greetings, yes/no confirmations, thanks, small talk — nothing needs a backend lookup.",
}


@dataclass
class ToolResult:
    tool: str
    ok: bool
    data: dict
    say: str            # a short natural-language summary the LLM (or mock LLM) can speak
    latency_ms: float
    timed_out: bool = False


class FakeClinicBackend:
    def __init__(self, rng: random.Random | None = None, p50_ms: float = 140, fail_rate: float = 0.0):
        self.rng = rng or random.Random(0)
        self.p50 = p50_ms
        self.fail_rate = fail_rate

    async def call(self, tool: str, query: str) -> dict:
        # lognormal latency with a fat tail, like real backends
        await sleep_ms(self.p50 * self.rng.lognormvariate(0, 0.45))
        if self.rng.random() < self.fail_rate:
            raise ConnectionError("backend unavailable")
        return _FAKE_DATA.get(tool, {})


_FAKE_DATA = {
    "lookup_appointment": {"date": "Tuesday", "time": "10:30 am", "clinic": "Lakeside Clinic",
                           "say": "Your next appointment is on Tuesday at ten thirty at Lakeside Clinic."},
    "reschedule_appointment": {"slots": ["Thursday 9 am", "Friday 2 pm"],
                               "say": "I can move it to Thursday at nine or Friday at two. Which works better?"},
    "medication_info": {"drug": "metformin", "dose": "500 mg", "schedule": "twice daily with meals",
                        "say": "Your metformin is five hundred milligrams, twice a day with meals. Your refill is ready at the pharmacy."},
    "report_side_effect": {"escalate": True,
                           "say": "Thank you for telling me. I'm flagging this for a nurse, who will call you back today."},
}

FALLBACK_SAY = {
    "lookup_appointment": "I'm having trouble pulling up your schedule right now. A team member will text you the details shortly.",
    "reschedule_appointment": "I can't reach the scheduling system right now. I'll have the front desk call you to reschedule.",
    "medication_info": "I can't access your medication record at the moment. A pharmacist will follow up with you today.",
    "report_side_effect": "I'm flagging this for a nurse right away, who will call you back today.",
}


class ToolRunner:
    def __init__(self, backend: FakeClinicBackend, timeout_ms: float = 800):
        self.backend = backend
        self.timeout_ms = timeout_ms

    async def run(self, tool: str, query: str) -> ToolResult | None:
        if not tool or tool == "none" or tool not in TOOLS:
            return None
        t0 = now_ms()
        try:
            data = await asyncio.wait_for(self.backend.call(tool, query), self.timeout_ms / 1000)
            return ToolResult(tool, True, data, data.get("say", ""), now_ms() - t0)
        except asyncio.TimeoutError:
            return ToolResult(tool, False, {}, FALLBACK_SAY[tool], now_ms() - t0, timed_out=True)
        except Exception:
            return ToolResult(tool, False, {}, FALLBACK_SAY[tool], now_ms() - t0)


def openai_tool_specs() -> list[dict]:
    """Same tools, in OpenAI function-calling format (for the LLM-routing baseline)."""
    return [{"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": []}}}
        for name, desc in TOOLS.items() if name != "none"]
