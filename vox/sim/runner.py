"""Run scripted calls through the orchestrator in virtual time and collect metrics."""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from pathlib import Path

from ..clock import run_virtual
from ..config import PRESETS, Strategy
from ..deciders import Decider, FallbackDecider, HeuristicDecider
from ..eval.metrics import _pct, per_turn
from ..orchestrator import Orchestrator
from ..phi import Redactor
from ..providers.mock import MockLLM, MockProfile, MockTTS, SimAudioOut
from ..tools import FakeClinicBackend, ToolRunner
from ..tracing import Tracer
from .player import Player
from .scenarios import SCENARIOS, Scenario

DECIDER_FREE = {"silence"}  # strategies that never call the decider


def build_decider(kind: str, rng: random.Random, cache_dir: Path, budget_ms: float, sim_latency_ms: float) -> Decider:
    if kind == "heuristic":
        return HeuristicDecider(simulated_latency_ms=sim_latency_ms, rng=rng)
    if kind == "none":
        return HeuristicDecider(0.0)
    if kind == "minijev":
        from ..deciders.minijev import MiniJevDecider
        return MiniJevDecider(sim_compute_ms=5.0)
    from ..deciders.bridge import SimBridge
    if kind == "jev":
        from ..deciders.jev import JevDecider
        inner = JevDecider()
    elif kind == "ollama":
        from ..deciders import make_ollama
        inner = make_ollama()
    elif kind == "ollaya":
        from ..deciders import make_ollaya
        inner = make_ollaya()
    elif kind == "llm":
        from ..deciders.llm_json import LLMJsonDecider
        inner = LLMJsonDecider()
    else:
        raise ValueError(kind)
    bridged = SimBridge(inner, cache_dir / f"decider_cache_{kind}.json")
    return FallbackDecider(bridged, HeuristicDecider(0.0), budget_ms=budget_ms)


def _uses_decider(s: Strategy) -> bool:
    return s.turn_mode == "semantic" or s.routing == "decider" or s.bargein == "semantic"


@dataclass
class RunResult:
    rows: list[dict]
    ov_rows: list[dict]
    decider: Decider
    orch: Orchestrator
    player: Player
    tracer: Tracer


def run_one(scenario: Scenario, variant: int, strategy: Strategy, decider_kind: str, *,
            profile: MockProfile | None = None, cache_dir: Path = Path("runs/cache"),
            budget_ms: float = 350, sim_latency_ms: float = 150, log=None) -> RunResult:
    seed = zlib.crc32(f"{scenario.name}:{variant}".encode())
    profile = profile or MockProfile()
    user_rng = random.Random(seed)
    prov_rng = random.Random(seed + 1)
    tool_rng = random.Random(seed + 2)
    dec_rng = random.Random(seed + 3)

    redactor = Redactor([scenario.patient])
    tracer = Tracer(redactor, conversation_id=f"{redactor.pseudonym(scenario.patient)}-{variant}")
    decider = build_decider(decider_kind if _uses_decider(strategy) else "none", dec_rng, cache_dir,
                            budget_ms, sim_latency_ms)

    async def main():
        player_ref: list[Player] = []
        llm = MockLLM(prov_rng, profile, oracle_tool=lambda: player_ref[0].current_tool)
        orch = Orchestrator(strategy, decider, llm, MockTTS(prov_rng, profile), SimAudioOut(),
                            ToolRunner(FakeClinicBackend(tool_rng)), tracer, log=log)
        player = Player(orch, scenario, user_rng)
        player_ref.append(player)
        await player.run()
        return orch, player

    orch, player = run_virtual(main())
    tags = {"strategy": strategy.name, "decider": decider_kind if _uses_decider(strategy) else "-",
            "scenario": scenario.name, "variant": variant}
    rows, ov_rows = per_turn(orch, player, tags)
    return RunResult(rows, ov_rows, decider, orch, player, tracer)


def bench(strategies: list[str], deciders: list[str], variants: int, *, cache_dir: Path,
          budget_ms: float = 350, sim_latency_ms: float = 150, trace_dir: Path | None = None,
          progress=print) -> tuple[list[dict], list[dict], dict[str, dict]]:
    rows, ov_rows, dstats = [], [], {}
    for sname in strategies:
        strat = PRESETS[sname]
        for dk in (deciders if _uses_decider(strat) else ["-"]):
            lat_all, calls, fallbacks = [], 0, 0
            for sc in SCENARIOS:
                for v in range(variants):
                    res = run_one(sc, v, strat, dk if dk != "-" else "none", cache_dir=cache_dir,
                                  budget_ms=budget_ms, sim_latency_ms=sim_latency_ms)
                    for r in res.rows + res.ov_rows:
                        r["decider"] = dk
                    rows += res.rows
                    ov_rows += res.ov_rows
                    st = res.decider.stats
                    lat_all += st.latencies_ms
                    calls += st.calls
                    fallbacks += st.fallbacks
                    if trace_dir and v == 0:
                        res.tracer.export(trace_dir, f"{sname}__{dk}__{sc.name}")
            dstats[f"{sname}|{dk}"] = {"calls": calls, "fallbacks": fallbacks,
                                       "p50": round(_pct(lat_all, 0.5)) if lat_all else None}
            progress(f"  ✓ {sname:<15} decider={dk:<10} {len(SCENARIOS) * variants} calls simulated")
    return rows, ov_rows, dstats
