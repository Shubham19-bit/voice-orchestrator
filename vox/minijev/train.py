"""Train and evaluate mini-Jev.

    python -m vox.minijev.train            # ~10-30 s on a laptop CPU

Evaluation uses two test sets:
  * synthetic held-out split (same generator, unseen sentences)  -> "does it learn?"
  * the hand-written benchmark scenarios (never used in training)  -> "does it generalise?"
and compares against the rule-based heuristic on the same examples.
"""
from __future__ import annotations

import random
import statistics
import time
from pathlib import Path

from ..deciders.heuristic import eot_probability, overlap_kind, route_tool, urgency
from ..sim.scenarios import SCENARIOS, parse_turn
from .data import eot_examples, overlap_examples, utterances
from .model import DEFAULT_PATH, MiniJev


def build_dataset(seed: int = 7, n_utts: int = 6000) -> tuple[dict, dict]:
    rng = random.Random(seed)
    utts = utterances(n_utts, rng)
    rng.shuffle(utts)
    cut = int(len(utts) * 0.85)
    tr, va = utts[:cut], utts[cut:]

    def pack(us, n_ov):
        ex, ey = eot_examples(us, rng)
        ox, oy = overlap_examples(us, n_ov, rng)
        return {"eot": (ex, ey),
                "tool": ([u[0] for u in us], [u[1] for u in us]),
                "urgency": ([u[0] for u in us], [u[2] for u in us]),
                "overlap": (ox, oy)}
    return pack(tr, 3000), pack(va, 600)


def scenario_testset() -> dict:
    """Ground truth from the hand-written scenarios (never seen in training)."""
    eot_x, eot_y, tool_x, tool_y, urg_x, urg_y, ov_x, ov_y = [], [], [], [], [], [], [], []
    for sc in SCENARIOS:
        for t in sc.turns:
            toks = parse_turn(t.text)
            words = [w for w, _ in toks]
            for i, (_, pause) in enumerate(toks[:-1]):
                if pause is not None:          # the patient paused here mid-turn -> NOT finished
                    eot_x.append(" ".join(words[: i + 1])); eot_y.append(0)
            full = " ".join(words)
            eot_x.append(full); eot_y.append(1)
            tool_x.append(full); tool_y.append(t.tool)
            urg_x.append(full); urg_y.append(t.urgency)
            if t.backchannel_at_ms is not None:
                ov_x.append(t.backchannel_text); ov_y.append("backchannel")
            if t.barge_in_at_ms is not None:  # what the decider sees 1, 2 and 3 words into an interruption
                for k in (1, 2, 3):
                    ov_x.append(" ".join(words[:k])); ov_y.append("interruption")
    return {"eot": (eot_x, eot_y), "tool": (tool_x, tool_y), "urgency": (urg_x, urg_y), "overlap": (ov_x, ov_y)}


def _acc(pred, truth) -> float:
    return sum(p == t for p, t in zip(pred, truth)) / max(1, len(truth))


def evaluate(m: MiniJev, ds: dict) -> dict:
    ex, ey = ds["eot"]
    tx, ty = ds["tool"]
    ux, uy = ds["urgency"]
    ox, oy = ds["overlap"]
    mj = {
        "eot": _acc([int(m.noul_eot(x)["noul"] >= 0.5) for x in ex], ey),
        "tool": _acc([m.choice_tool(x)["choice"] for x in tx], ty),
        "urgency": _acc([round(m.score_urgency(x)["score"]) for x in ux], uy),
        "overlap": _acc([m.choice_overlap(x)["choice"] for x in ox], oy),
    }
    heur = {
        "eot": _acc([int(eot_probability(x) >= 0.5) for x in ex], ey),
        "tool": _acc([route_tool(x)[0] for x in tx], ty),
        "urgency": _acc([round(urgency(x)) for x in ux], uy),
        "overlap": _acc([overlap_kind(x)[0] for x in ox], oy),
    }
    n = {k: len(ds[k][1]) for k in ds}
    return {"minijev": mj, "heuristic": heur, "n": n}


def latency_ms(m: MiniJev, n: int = 300) -> tuple[float, float]:
    texts = ["I wanted to ask about my um", "can we move it to friday", "and last night I had chest pain",
             "okay thank you", "so I forgot to take my lisinopril yesterday"]
    lat = []
    for i in range(n):
        t = texts[i % len(texts)]
        t0 = time.perf_counter()
        m.noul_eot(t); m.choice_tool(t); m.score_urgency(t)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    return statistics.median(lat), lat[int(0.95 * (n - 1))]


def report(syn: dict, real: dict, lat: tuple[float, float], train_s: float, sizes: dict) -> str:
    rows = ["| Decision | mini-Jev (synthetic test) | mini-Jev (real scenarios) | Heuristic (real scenarios) | n real |",
            "|---|---|---|---|---|"]
    names = {"eot": "End of turn (noul)", "tool": "Tool routing (choice)", "urgency": "Urgency (score)",
             "overlap": "Backchannel vs interruption (choice)"}
    for k, label in names.items():
        rows.append(f"| {label} | {syn['minijev'][k]:.0%} | **{real['minijev'][k]:.0%}** | "
                    f"{real['heuristic'][k]:.0%} | {real['n'][k]} |")
    return "\n".join([
        "# mini-Jev training report", "",
        f"- training examples: " + ", ".join(f"{k}={v:,}" for k, v in sizes.items()),
        f"- training time: {train_s:.1f} s (CPU)",
        f"- inference latency (eot + tool + urgency together): p50 {lat[0]:.2f} ms, p95 {lat[1]:.2f} ms", "",
        *rows, "",
        "Real scenarios are the hand-written benchmark calls (never used for training). They are small,",
        "so treat differences of one or two examples with care.",
    ])


def main() -> None:
    print("Generating synthetic data…")
    train, val = build_dataset()
    sizes = {k: len(v[1]) for k, v in train.items()}
    print("  " + ", ".join(f"{k}={v:,}" for k, v in sizes.items()))
    print("Training 4 heads…")
    t0 = time.time()
    m = MiniJev().fit(train)
    train_s = time.time() - t0
    m.meta = {"sizes": sizes, "train_s": train_s}
    path = m.save()
    syn, real = evaluate(m, val), evaluate(m, scenario_testset())
    lat = latency_ms(m)
    md = report(syn, real, lat, train_s, sizes)
    (path.parent / "minijev_report.md").write_text(md, encoding="utf-8")
    print("\n" + md + f"\n\nSaved model → {path}")


if __name__ == "__main__":
    main()
