"""Synthetic training data for mini-Jev.

We generate patient utterances from templates (5 intents x urgency levels), then
derive the four decision datasets from them:

  eot      : (partial transcript) -> finished turn?  [complete utterance = 1,
             utterance chopped mid-way / ending in a filler = 0]
  tool     : (utterance) -> which backend tool
  urgency  : (utterance) -> 0 routine / 1 follow-up / 2 urgent
  overlap  : (what the patient said over the agent) -> backchannel / interruption / other

IMPORTANT: the evaluation set (vox/sim/scenarios.py) is hand-written separately and
is never used here — we also drop any generated sentence that collides with it.
"""
from __future__ import annotations

import random
import re

MEDS = ["metformin", "lisinopril", "atorvastatin", "amlodipine", "insulin", "levothyroxine", "losartan",
        "omeprazole", "the blood pressure pills", "the new tablets", "my inhaler", "the statin",
        "my diabetes medicine", "the water pill", "my heart medication", "the antibiotics"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "next week", "tomorrow",
        "the twelfth", "the end of the month", "this week", "the weekend"]
TIMES = ["in the morning", "in the afternoon", "after lunch", "around ten", "at nine", "later in the day",
         "before noon", "after five"]
TIMINGS = ["with food", "on an empty stomach", "at night", "in the morning", "twice a day", "with my other pills",
           "before bed", "after dinner"]
MILD = ["a bit dizzy", "nauseous", "really tired", "a little sick", "kind of off", "achy", "a headache",
        "an upset stomach", "a rash on my arm", "itchy", "a dry cough", "some muscle pain", "lightheaded"]
SEVERE = ["chest pain", "trouble breathing", "my throat swelling", "really bad chest tightness",
          "shortness of breath", "fainted this morning", "my face is swelling up", "severe pain in my chest",
          "I can't breathe properly", "I passed out", "blood in my urine", "my heart racing really fast"]

TEMPLATES: dict[str, list[str]] = {
    "medication_info": [
        "can I get a refill on {med}", "I need a refill for {med}", "I'm running low on {med}",
        "how many {med} should I take", "should I take {med} {timing}", "is it okay to take {med} {timing}",
        "I missed my dose of {med} yesterday", "I forgot to take {med} this morning what should I do",
        "what's the dose for {med}", "can I take {med} with my other pills", "I ran out of {med}",
        "has my prescription for {med} been sent to the pharmacy", "which pharmacy did you send {med} to",
        "do I keep taking {med} or stop", "can I split the {med} tablets", "is the refill for {med} ready",
        "I have a question about {med}", "my pharmacy says {med} isn't ready yet",
        "how long do I stay on {med}", "can I double up on {med} if I missed one",
    ],
    "reschedule_appointment": [
        "can we move my appointment to {day}", "I need to reschedule my appointment", "I can't make it on {day}",
        "can I come in {time} instead", "is there anything available {day}", "I need to cancel my appointment",
        "can we push it to {day}", "could I switch to {day} {time}", "is there an earlier slot",
        "something came up so I need a different day", "can I book a follow up for {day}",
        "{day} {time} works better for me", "I'd like to change my visit to {day}",
        "can you move it to {time}", "I have to work on {day} so can we change it",
        "can I get a later time", "please cancel the visit on {day}",
    ],
    "lookup_appointment": [
        "when is my next appointment", "what time is my appointment on {day}", "which clinic am I going to",
        "can you remind me when my visit is", "is my appointment still on for {day}",
        "where do I go for my appointment", "who am I seeing at my next visit",
        "do I have anything booked for {day}", "what day was my checkup again", "is my blood test {day}",
        "do I need to fast before my appointment", "what should I bring to my visit",
        "can you confirm my appointment", "I just want to check my appointment time",
        "is it with the same doctor as last time",
    ],
    "report_side_effect": [
        "I've been feeling {mild} since I started {med}", "{med} is making me feel {mild}",
        "I get {mild} after taking {med}", "ever since the new dose I've had {mild}",
        "I think {med} is giving me {mild}", "I've had {mild} for a few days now",
        "I feel {mild} every morning", "I noticed {mild} after my last dose",
    ],
    "none": [
        "hello", "hi", "yes", "yeah", "yes speaking", "this is she", "this is he", "speaking", "who is this",
        "who's calling", "okay", "okay thanks", "thank you", "thanks so much", "that's all", "no that's it",
        "nothing else", "bye", "goodbye", "sounds good", "perfect", "great", "alright", "sure", "yes please",
        "no thank you", "sorry can you repeat that", "what did you say", "I didn't catch that",
        "can you say that again", "that works", "got it", "no", "nope", "I'm doing well", "I'm fine thanks",
        "I'm good how are you", "is this the clinic", "hold on a second", "one moment please", "yes that's right",
        "correct", "that's correct", "no that's wrong", "which day was that", "what time was that",
        "sorry which one", "what was the name again", "can you spell that", "who am I speaking with",
        "hi there", "hello yes", "yes it is", "hey", "good morning", "good afternoon", "that's me",
        "yes this is her", "uh huh that's right", "okay perfect", "wonderful thank you", "alright thanks",
    ],
}
NAMES = ["anita sharma", "john miller", "rahul verma", "sara lee", "david kim", "fatima noor", "li wei",
         "carlos diaz", "emma brown", "arjun nair", "grace okafor", "tom wilson", "meera iyer", "ali hassan"]
TEMPLATES["none"] += ["this is {name}", "yes this is {name}", "{name} speaking", "it's {name}",
                      "hi this is {name}", "my name is {name}", "you're speaking with {name}"]
DISFLUENCIES = ["um", "uh", "like", "you know", "I mean", "erm"]

SEVERE_TEMPLATES = [
    "I've had {severe} since last night", "I'm having {severe}", "last night I had {severe}",
    "I took {med} and now I have {severe}", "after {med} I got {severe}", "I think I need help I have {severe}",
    "since this morning I've had {severe}", "I woke up with {severe}",
]
PREFIXES = ["", "", "", "", "um ", "uh ", "so ", "okay so ", "yeah ", "hi ", "actually ", "well ", "sorry ",
            "um so ", "yes ", "hello ", "and ", "also "]
SUFFIXES = ["", "", "", "", "", "", "", " please", " thanks", " I think", " if that's okay", " please thank you",
            " right", " again", " because I have to work", " since yesterday", " this week", " if possible",
            " for my mom", " this morning", " last week", " when you get a chance", " as soon as possible",
            " because I'm traveling", " if there's room", " for the follow up"]
FILLERS = ["um", "uh", "erm", "like", "and", "so", "but", "you know", "and um", "so uh", "I mean", "because"]

BACKCHANNELS = ["mm-hmm", "mhm", "uh-huh", "uh huh", "okay", "ok", "yeah", "yes", "right", "sure", "got it",
                "I see", "alright", "yep", "mm", "okay okay", "yeah yeah", "right right", "uh-huh okay",
                "oh okay", "ah", "oh", "hmm", "yes yes", "okay yeah", "sure sure", "oh I see"]
INTERRUPT_STARTS = ["wait", "sorry", "hold on", "actually", "no no", "excuse me", "hang on", "stop",
                    "sorry what", "wait wait", "no", "but", "one second", "can you repeat", "what was that",
                    "sorry which", "hold on hold on", "um sorry", "excuse me but", "wait what"]
SIDE_TALK = ["hey mark dinner's ready", "one sec honey", "*cough*", "cough cough", "turn that down",
             "I'm on the phone", "not now", "go ask your dad", "be right there", "shh", "achoo",
             "hey can you get the door", "sorry kids are loud", "put that down"]


def _fill(t: str, rng: random.Random) -> str:
    return (t.replace("{med}", rng.choice(MEDS)).replace("{day}", rng.choice(DAYS))
             .replace("{time}", rng.choice(TIMES)).replace("{timing}", rng.choice(TIMINGS))
             .replace("{mild}", rng.choice(MILD)).replace("{severe}", rng.choice(SEVERE))
             .replace("{name}", rng.choice(NAMES)))


def _disfluent(text: str, rng: random.Random) -> str:
    """Real speech is messy: drop a filler somewhere in the MIDDLE of a complete sentence."""
    w = text.split()
    if len(w) >= 3 and rng.random() < 0.3:
        w.insert(rng.randint(1, len(w) - 1), rng.choice(DISFLUENCIES))
    return " ".join(w)


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'\-]", " ", t.lower())).strip()


def _excluded() -> set[str]:
    from ..sim.scenarios import SCENARIOS, parse_turn
    # Ban every benchmark sentence of 4+ words. Shorter ones ("hello", "okay thank you") are
    # stock phrases any system must know, so banning them would just cripple the model.
    texts = {_norm(" ".join(w for w, _ in parse_turn(t.text))) for s in SCENARIOS for t in s.turns}
    return {t for t in texts if len(t.split()) >= 4}


def utterances(n: int, rng: random.Random) -> list[tuple[str, str, int, int]]:
    """-> [(text, tool, urgency, n_prefix_words)]"""
    banned = _excluded()
    out: list[tuple[str, str, int]] = []
    tools = list(TEMPLATES)
    while len(out) < n:
        r = rng.random()
        if r < 0.10:
            tool, urg, t = "report_side_effect", 2, rng.choice(SEVERE_TEMPLATES)
        else:
            tool = "none" if rng.random() < 0.25 else rng.choice(tools)
            t = rng.choice(TEMPLATES[tool])
            urg = 1 if tool == "report_side_effect" else 0
        prefix = rng.choice(PREFIXES)
        text = _norm(prefix + _disfluent(_fill(t, rng), rng) + rng.choice(SUFFIXES))
        if text and text not in banned:
            out.append((text, tool, urg, len(prefix.split())))
    return out


def eot_examples(utts: list[tuple[str, str, int, int]], rng: random.Random) -> tuple[list[str], list[int]]:
    """Complete utterances -> 1. Chopped prefixes / trailing fillers -> 0.

    We only cut INSIDE the request itself, never right after an opener like "hi" / "yes" /
    "okay so" — a pause after "hi" is genuinely ambiguous, and labelling it "not done"
    thousands of times teaches the model that greetings are never finished."""
    X, y = [], []
    complete = {u[0] for u in utts}
    standalone = {_norm(t) for t in TEMPLATES["none"] if "{" not in t}
    openers = {"um", "uh", "so", "okay", "yeah", "hi", "actually", "well", "sorry", "yes", "hello", "and", "also"}

    def ambiguous(pre: str) -> bool:
        """A cut that is itself something people say as a whole turn ("hi", "yes", "okay so hi")."""
        w = pre.split()
        while w and w[0] in openers and " ".join(w) not in standalone:
            w = w[1:]
        return not w or pre in complete or " ".join(w) in standalone
    for text, _, _, npre in utts:
        words = text.split()
        X.append(text); y.append(1)
        if len(words) - npre >= 2:
            for _ in range(2):  # two random cut points per utterance
                k = rng.randint(npre + 1, len(words) - 1)
                pre = " ".join(words[:k])
                if ambiguous(pre):
                    continue  # the prefix is itself a complete utterance somewhere -> skip (ambiguous)
                X.append(pre); y.append(0)
        if rng.random() < 0.5:  # thinking out loud: "... my um", "... and"
            k = rng.randint(1, len(words))
            X.append(" ".join(words[:k] + [rng.choice(FILLERS)])); y.append(0)
        if rng.random() < 0.15:  # complete, with a trailing politeness word still complete
            X.append(text + " " + rng.choice(["please", "thanks", "thank you"])); y.append(1)
    return X, y


def overlap_examples(utts: list[tuple[str, str, int, int]], n: int, rng: random.Random) -> tuple[list[str], list[str]]:
    X, y = [], []
    for _ in range(n):
        r = rng.random()
        if r < 0.40:
            X.append(rng.choice(BACKCHANNELS)); y.append("backchannel")
        elif r < 0.85:
            if rng.random() < 0.5:
                start = rng.choice(INTERRUPT_STARTS)
                rest = rng.choice(utts)[0].split()
                text = start + " " + " ".join(rest[: rng.randint(0, min(5, len(rest)))])
            else:  # the patient just starts a new request over the agent (first few words only)
                words = rng.choice([u for u in utts if u[1] != "none"])[0].split()
                text = " ".join(words[: rng.randint(2, min(6, len(words)))])
            X.append(_norm(text)); y.append("interruption")
        else:
            X.append(_norm(rng.choice(SIDE_TALK)) or "cough"); y.append("other")
    return X, y
