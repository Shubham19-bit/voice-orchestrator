"""Command line entry point.

  python -m vox.cli demo      [--scenario refill] [--strategy hybrid] [--decider heuristic]
  python -m vox.cli bench     [--deciders heuristic,jev] [--variants 5]
  python -m vox.cli probe     [--decider jev] [-n 20]          # measure real decider latency
  python -m vox.cli live      [--strategy hybrid] [--decider jev]   # talk to it (needs keys + mic)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path


def load_env(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# --------------------------------------------------------------------------- demo
def cmd_demo(a) -> None:
    from .config import PRESETS
    from .sim.runner import run_one
    from .sim.scenarios import SCENARIOS

    sc = next((s for s in SCENARIOS if s.name == a.scenario), None)
    if sc is None:
        sys.exit(f"unknown scenario; choose from {[s.name for s in SCENARIOS]}")
    res = run_one(sc, a.variant, PRESETS[a.strategy], a.decider, cache_dir=Path(a.out) / "cache")
    t0 = res.player.turns[0].start
    print(f"\n=== {sc.name} · strategy={a.strategy} · decider={a.decider}  (times in ms, virtual clock)\n")
    interesting = {"decide_turn", "decide_overlap", "response_start", "commit", "barge_in_stop", "playback",
                   "response_cancelled_spec", "response_cancelled_late", "response_stale"}
    events = [("user", t.start, f"🧑 patient: \"{t.text}\"  (speaks {t.end - t.start:.0f} ms)") for t in res.player.turns]
    events += [("user", b.start, f"🧑 (backchannel) \"{b.text}\"") for b in res.player.backchannels]
    for s in res.tracer.spans:
        if s.name not in interesting:
            continue
        at = s.attrs
        if s.name == "decide_turn":
            msg = f"   ⚖ decide  P(done)={at.get('eot')} tool={at.get('tool')} [{s.dur_ms:.0f} ms]  \"{at.get('text')}\""
        elif s.name == "decide_overlap":
            msg = f"   ⚖ overlap → {at.get('kind')} [{s.dur_ms:.0f} ms]  \"{at.get('text')}\""
        elif s.name == "playback":
            msg = f"   🔊 agent speaks for {s.dur_ms:.0f} ms"
        elif s.name == "response_start":
            msg = f"   ▶ pipeline start #{at.get('response')} {'(speculative)' if at.get('speculative') else ''}"
        else:
            msg = f"   • {s.name.replace('_', ' ')} #{at.get('response', '')} {at.get('after_ms', '')}"
        events.append(("sys", s.start_ms, msg))
    for _, t, msg in sorted(events, key=lambda e: e[1]):
        print(f"{t - t0:8.0f}  {msg}")
    print("\nPer-turn results:")
    for r in res.rows:
        lat = f"{r['latency_ms']:.0f} ms" if r.get("latency_ms") is not None else "no reply"
        print(f"  turn {r['turn']}: latency {lat:<9} cut-in={r['cut_in']!s:<5} tool={r.get('tool')} "
              f"(truth {r['truth_tool']})  \"{r['text']}\"")
    out = Path(a.out) / "demo"
    res.tracer.export(out, f"{sc.name}__{a.strategy}")
    print(f"\nTrace written to {out}/ — drag the .perfetto.json into https://ui.perfetto.dev\n")


# --------------------------------------------------------------------------- bench
def cmd_bench(a) -> None:
    from .eval.metrics import summarize
    from .eval.report import markdown, write_report
    from .sim.runner import bench
    from .sim.scenarios import SCENARIOS

    deciders = [d.strip() for d in a.deciders.split(",") if d.strip()]
    for d in deciders:
        if d == "jev" and not os.environ.get("TYPESAFE_API_KEY"):
            sys.exit("--deciders jev needs TYPESAFE_API_KEY in .env")
        if d == "llm" and not os.environ.get("LLM_API_KEY"):
            sys.exit("--deciders llm needs LLM_API_KEY in .env")
    out = Path(a.out) / time.strftime("bench-%Y%m%d-%H%M%S")
    print(f"Running {len(SCENARIOS)} scenarios × {a.variants} variants per configuration…")
    t = time.time()
    rows, ov, ds = bench(a.strategies.split(","), deciders, a.variants, cache_dir=Path(a.out) / "cache",
                         budget_ms=a.budget_ms, sim_latency_ms=a.sim_latency_ms, trace_dir=out / "traces")
    summary = summarize(rows, ov, ds)
    report = write_report(out, rows, ov, summary, {
        "calls": len(SCENARIOS) * a.variants, "turns": sum(len(s.turns) for s in SCENARIOS) * a.variants,
        "deciders": deciders, "budget_ms": a.budget_ms})
    print(f"\n{markdown(summary)}\n\nDone in {time.time() - t:.1f}s → {report}")


# --------------------------------------------------------------------------- probe
def cmd_probe(a) -> None:
    from .deciders import make_decider

    texts = ["I wanted to ask about my um", "I wanted to ask about my metformin refill",
             "and last night I had some chest pain and trouble breathing", "can we move it to Friday",
             "okay thank you", "so I forgot to take my"]

    async def main():
        d = make_decider(a.decider, budget_ms=10_000)
        lats, fb = [], 0
        for i in range(a.n):
            txt = texts[i % len(texts)]
            r = await d.analyze_turn(txt)
            lats.append(r.latency_ms)
            fb += r.fallback
            print(f"  {r.latency_ms:6.0f} ms  P(done)={r.eot_prob:.2f} tool={r.tool:<22} urg={r.urgency:.1f} "
                  f"{'(FALLBACK)' if r.fallback else ''} {txt!r}")
        lats.sort()
        print(f"\n{a.decider}: n={len(lats)} p50={statistics.median(lats):.0f} ms "
              f"p90={lats[int(0.9 * (len(lats) - 1))]:.0f} ms max={lats[-1]:.0f} ms fallbacks={fb}")
        print("Note: the first call includes TLS connection setup; keep connections warm in production.")
        from .http import aclose_all
        await aclose_all()

    asyncio.run(main())


# --------------------------------------------------------------------------- live
def cmd_live(a) -> None:
    from .live import run_live
    asyncio.run(run_live(a.strategy, a.decider, Path(a.out) / "live"))


def main(argv=None) -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252; we print emoji
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    load_env()
    p = argparse.ArgumentParser(prog="vox", description="Latency-optimised voice agent orchestration")
    p.add_argument("--out", default="runs")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="replay one scripted call and print the timeline")
    d.add_argument("--scenario", default="refill")
    d.add_argument("--strategy", default="hybrid-v2")
    d.add_argument("--decider", default="heuristic", help="heuristic | minijev | ollama | ollaya | llm | jev")
    d.add_argument("--variant", type=int, default=0)
    d.set_defaults(fn=cmd_demo)

    b = sub.add_parser("bench", help="run all scenarios × strategies, write report + charts")
    b.add_argument("--strategies", default="baseline-700,aggressive-400,semantic-eot,hybrid,hybrid-v2")
    b.add_argument("--deciders", default="heuristic", help="comma list: heuristic,minijev,ollama,ollaya,llm,jev")
    b.add_argument("--variants", type=int, default=5)
    b.add_argument("--budget-ms", type=float, default=350, help="decider latency budget before fallback")
    b.add_argument("--sim-latency-ms", type=float, default=150, help="simulated network latency for 'heuristic'")
    b.set_defaults(fn=cmd_bench)

    pr = sub.add_parser("probe", help="measure a real decider's latency")
    pr.add_argument("--decider", default="ollama")
    pr.add_argument("-n", type=int, default=20)
    pr.set_defaults(fn=cmd_probe)

    lv = sub.add_parser("live", help="talk to the agent through your mic (headphones!)")
    lv.add_argument("--strategy", default="hybrid-v2")
    lv.add_argument("--decider", default="heuristic-local", help="heuristic-local | minijev | ollama | ollaya | llm | jev")
    lv.set_defaults(fn=cmd_live)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
