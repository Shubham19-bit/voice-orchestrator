from __future__ import annotations

from .base import Decider, OverlapAnalysis, TurnAnalysis
from .fallback import FallbackDecider
from .heuristic import HeuristicDecider


def make_decider(kind: str, *, budget_ms: float = 350, sim_latency_ms: float = 150, rng=None) -> Decider:
    """kind: heuristic | heuristic-local | minijev | ollama | ollaya | llm | jev"""
    if kind == "heuristic":
        # pretend it's a remote call with ~Jev-like latency, for fair comparisons
        return HeuristicDecider(simulated_latency_ms=sim_latency_ms, rng=rng)
    if kind == "heuristic-local":
        return HeuristicDecider(0.0)
    if kind == "jev":
        from .jev import JevDecider
        return FallbackDecider(JevDecider(), HeuristicDecider(0.0), budget_ms=budget_ms)
    if kind == "ollaya":
        return FallbackDecider(make_ollaya(), HeuristicDecider(0.0), budget_ms=budget_ms)
    if kind == "minijev":
        from .minijev import MiniJevDecider
        return MiniJevDecider()
    if kind == "ollama":
        return FallbackDecider(make_ollama(), HeuristicDecider(0.0), budget_ms=budget_ms)
    if kind == "llm":
        from .llm_json import LLMJsonDecider
        return FallbackDecider(LLMJsonDecider(), HeuristicDecider(0.0), budget_ms=budget_ms)
    raise ValueError(f"unknown decider {kind!r}")


__all__ = ["Decider", "TurnAnalysis", "OverlapAnalysis", "HeuristicDecider", "FallbackDecider", "make_decider"]


def make_ollaya():
    """Free, local, Jev-compatible decider: Ollaya serves the same /v1/systemone wire format.

    Install: https://ollaya.dev  ·  run `ollaya serve`  ·  default model laya:en (small encoder, runs on CPU)
    """
    import os
    from .jev import JevDecider
    d = JevDecider(api_key="local",
                   model=os.environ.get("OLLAYA_MODEL", "laya:en"),
                   url=os.environ.get("OLLAYA_URL", "http://127.0.0.1:11435/v1/systemone"))
    d.name = "ollaya"
    return d


def make_ollama():
    """Free, local decider via Ollama (ollama.com) — a small open model answering in JSON mode.

    Setup:  ollama pull qwen2.5:1.5b      (then Ollama serves http://localhost:11434 in the background)
    """
    import os
    from .llm_json import LLMJsonDecider
    d = LLMJsonDecider(api_key="ollama",
                       base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434/v1"),
                       model=os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b"))
    d.name = "ollama"
    return d
