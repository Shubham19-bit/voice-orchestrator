"""Synthetic patient calls (no real PHI). Pauses are written inline as <ms>.

The scripts are deliberately hard in the ways real calls are hard:
  * mid-sentence pauses after fillers / conjunctions ("my <600> um <500> metformin")
  * pauses after phrases that LOOK complete ("I'm not sure <1200> when my next ...")
  * backchannels while the agent talks ("mm-hmm")
  * real interruptions ("wait, sorry, which day was that?")
  * urgent symptoms that must be escalated

Each turn has ground truth for the tool it needs and its urgency (0-2).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class TurnSpec:
    text: str
    tool: str = "none"
    urgency: int = 0
    barge_in_at_ms: float | None = None      # this turn interrupts the agent's previous reply, N ms in
    backchannel_at_ms: float | None = None   # patient says a backchannel N ms into the agent's reply to this turn
    backchannel_text: str = "mm-hmm"


@dataclass
class Scenario:
    name: str
    patient: str
    turns: list[TurnSpec] = field(default_factory=list)


def parse_turn(text: str) -> list[tuple[str, float | None]]:
    """'I take <600> um it' -> [('I', None), ('take', 600), ('um', None), ('it', None)]"""
    out: list[tuple[str, float | None]] = []
    for tok in re.findall(r"<\d+>|[^\s<>]+", text):
        if tok.startswith("<"):
            if out:
                out[-1] = (out[-1][0], float(tok[1:-1]))
        else:
            out.append((tok, None))
    return out


SCENARIOS: list[Scenario] = [
    Scenario("refill", "Priya Raman", [
        TurnSpec("hi <300> yes this is Priya Raman", "none"),
        TurnSpec("I wanted to ask about my <600> um <500> metformin refill", "medication_info",
                 backchannel_at_ms=1400),
        TurnSpec("okay and should I take it before <750> or after meals", "medication_info"),
        TurnSpec("great thank you so much", "none"),
    ]),
    Scenario("reschedule", "Robert Chen", [
        TurnSpec("hello", "none"),
        TurnSpec("I need to move my appointment on <700> Tuesday because I have to work", "reschedule_appointment"),
        TurnSpec("wait <250> sorry which day was that", "none", barge_in_at_ms=1500),
        TurnSpec("okay Friday at two works for me", "reschedule_appointment", backchannel_at_ms=1200,
                 backchannel_text="okay"),
        TurnSpec("no that's all bye", "none"),
    ]),
    Scenario("side-effect", "Maria Gonzalez", [
        TurnSpec("yes speaking", "none"),
        TurnSpec("I've been feeling <500> really dizzy since I started the new pills", "report_side_effect", 1),
        TurnSpec("and last night I had some chest pain <450> and trouble breathing", "report_side_effect", 2),
        TurnSpec("okay thank you", "none"),
    ]),
    Scenario("elderly-long-pauses", "Harold Smith", [
        TurnSpec("hello <400> who is this", "none"),
        TurnSpec("um <900> I'm not sure <1200> when my next <800> appointment is", "lookup_appointment",
                 backchannel_at_ms=1600, backchannel_text="uh-huh"),
        TurnSpec("and is that with <1100> doctor Patel or the other one", "lookup_appointment"),
        TurnSpec("alright <600> thank you dear", "none"),
    ]),
    Scenario("missed-dose", "Aisha Khan", [
        TurnSpec("hi", "none"),
        TurnSpec("so I forgot to take my <650> lisinopril yesterday", "medication_info", backchannel_at_ms=1000,
                 backchannel_text="yeah"),
        TurnSpec("should I take two today <500> or just skip it", "medication_info"),
        TurnSpec("hold on <300> sorry can you repeat that", "none", barge_in_at_ms=1800),
        TurnSpec("got it thanks", "none"),
    ]),
    Scenario("confirm-visit", "James O'Brien", [
        TurnSpec("yeah hi", "none"),
        TurnSpec("I just wanted to confirm my appointment for <550> next week", "lookup_appointment"),
        TurnSpec("perfect <400> and do I need to fast before the blood test", "lookup_appointment"),
        TurnSpec("actually <300> can we move it to the afternoon", "reschedule_appointment", barge_in_at_ms=2200),
        TurnSpec("thanks bye", "none"),
    ]),
]
