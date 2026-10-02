"""Turn the raw timeline (orchestrator records + player ground truth) into metrics."""
from __future__ import annotations

import statistics
from collections import defaultdict

from ..orchestrator import Orchestrator
from ..sim.player import Player


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def per_turn(orch: Orchestrator, player: Player, tags: dict) -> tuple[list[dict], list[dict]]:
    turns = player.turns
    rows = []
    audible = [r for r in orch.responses if r.t_audio_start is not None]
    for i, t in enumerate(turns):
        nxt = turns[i + 1].start if i + 1 < len(turns) else float("inf")
        reply = next((r for r in audible if t.end <= r.t_audio_start < nxt), None)
        cutin = any(t.start < r.t_audio_start < t.end for r in audible)
        wasted = sum(1 for r in orch.responses
                     if r.outcome in ("cancelled_spec", "cancelled_late", "stale") and t.start <= r.t_trigger < nxt)
        row = {**tags, "turn": i, "text": t.text, "truth_tool": t.tool, "truth_urgency": t.urgency,
               "is_barge_in": t.is_barge_in, "replied": reply is not None, "cut_in": cutin,
               "wasted_runs": wasted}
        if reply:
            row.update(
                latency_ms=reply.t_audio_start - t.end,
                tool=reply.tool, tool_correct=reply.tool == t.tool,
                urgency_pred=(reply.analysis.urgency if reply.analysis else None),
                speculative=reply.speculative,
                wait_eot_ms=max(0.0, reply.t_trigger - t.end),
                route_ms=(reply.t_route_end - reply.t_trigger) if reply.t_route_end else 0.0,
                tool_ms=(reply.t_tool_end - reply.t_tool_start) if reply.t_tool_start else 0.0,
                llm_ttft_ms=(reply.t_llm_first - reply.t_llm_start) if reply.t_llm_first else 0.0,
                tts_ms=(reply.t_ready - reply.t_llm_first) if reply.t_llm_first else 0.0,
                gate_hold_ms=reply.t_audio_start - reply.t_ready,
            )
            # when the pipeline started *before* the turn ended (it can't for the real reply,
            # but keep the breakdown honest if it ever does)
        rows.append(row)

    ov_rows = []
    for bc in player.backchannels:
        ov = next((o for o in orch.overlaps if bc.start - 50 <= o.t_start <= bc.start + 300), None)
        if ov:
            ov_rows.append({**tags, "type": "backchannel", "text": bc.text, "stopped": ov.stopped,
                            "stop_ms": (ov.t_stop - bc.start) if ov.stopped else None})
    for t in turns:
        if t.is_barge_in and t.overlapped_agent:
            ov = next((o for o in orch.overlaps if t.start - 50 <= o.t_start <= t.start + 300), None)
            ov_rows.append({**tags, "type": "interruption", "text": t.text, "stopped": bool(ov and ov.stopped),
                            "stop_ms": (ov.t_stop - t.start) if ov and ov.stopped else None})
    return rows, ov_rows


def summarize(rows: list[dict], ov_rows: list[dict], decider_stats: dict[str, dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["strategy"], r["decider"])].append(r)
    ovg: dict[tuple, list[dict]] = defaultdict(list)
    for o in ov_rows:
        ovg[(o["strategy"], o["decider"])].append(o)

    out = []
    for key, rs in groups.items():
        lat = [r["latency_ms"] for r in rs if r.get("latency_ms") is not None]
        replied = [r for r in rs if r["replied"]]
        bcs = [o for o in ovg[key] if o["type"] == "backchannel"]
        ints = [o for o in ovg[key] if o["type"] == "interruption"]
        stop_lat = [o["stop_ms"] for o in ints if o["stop_ms"] is not None]
        urg = [r for r in replied if r.get("urgency_pred") is not None]
        ds = decider_stats.get("|".join(key), {})
        out.append({
            "strategy": key[0], "decider": key[1], "turns": len(rs),
            "latency_p50_ms": round(_pct(lat, 0.5)), "latency_p90_ms": round(_pct(lat, 0.9)),
            "latency_mean_ms": round(statistics.fmean(lat)) if lat else None,
            "cut_in_rate": round(sum(r["cut_in"] for r in rs) / len(rs), 3),
            "no_reply_rate": round(sum(not r["replied"] for r in rs) / len(rs), 3),
            "tool_accuracy": round(sum(r["tool_correct"] for r in replied) / len(replied), 3) if replied else None,
            "urgent_recall": (round(sum(1 for r in urg if r["truth_urgency"] == 2 and r["urgency_pred"] >= 1.5)
                                    / max(1, sum(1 for r in urg if r["truth_urgency"] == 2)), 3)
                              if any(r["truth_urgency"] == 2 for r in urg) else None),
            "wasted_runs_per_turn": round(sum(r["wasted_runs"] for r in rs) / len(rs), 3),
            "backchannel_false_stop_rate": round(sum(o["stopped"] for o in bcs) / len(bcs), 3) if bcs else None,
            "interruption_stop_rate": round(sum(o["stopped"] for o in ints) / len(ints), 3) if ints else None,
            "interruption_stop_p50_ms": round(_pct(stop_lat, 0.5)) if stop_lat else None,
            "decider_calls": ds.get("calls"), "decider_p50_ms": ds.get("p50"), "decider_fallbacks": ds.get("fallbacks"),
            **{f"avg_{k}": round(statistics.fmean([r[k] for r in replied if k in r]), 1) if replied else None
               for k in ("wait_eot_ms", "route_ms", "tool_ms", "llm_ttft_ms", "tts_ms", "gate_hold_ms")},
        })
    return out
