"""Write results: CSVs, a markdown summary and PNG charts."""
from __future__ import annotations

import csv
from pathlib import Path

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

STAGES = [("avg_wait_eot_ms", "waiting for end-of-turn"), ("avg_route_ms", "routing (LLM call or decider)"),
          ("avg_tool_ms", "backend tool"), ("avg_llm_ttft_ms", "LLM first token"),
          ("avg_tts_ms", "TTS first audio"), ("avg_gate_hold_ms", "held at speculation gate")]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _label(s: dict) -> str:
    return s["strategy"] + ("" if s["decider"] == "-" else f"\n({s['decider']})")


def _fmt(v, pct=False):
    if v is None:
        return "–"
    return f"{v * 100:.0f}%" if pct else f"{v:,}"


def markdown(summary: list[dict]) -> str:
    cols = [("strategy", "Strategy", None), ("decider", "Decider", None),
            ("latency_p50_ms", "Latency p50 (ms)", False), ("latency_p90_ms", "p90 (ms)", False),
            ("cut_in_rate", "Audible cut-ins", True), ("tool_accuracy", "Tool accuracy", True),
            ("backchannel_false_stop_rate", "Backchannel false stops", True),
            ("interruption_stop_p50_ms", "Interruption stop p50 (ms)", False),
            ("wasted_runs_per_turn", "Wasted pipeline runs / turn", None)]
    lines = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    for s in summary:
        cells = []
        for k, _, pct in cols:
            v = s.get(k)
            cells.append(str(v) if pct is None else _fmt(v, pct))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def charts(summary: list[dict], out: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    paths = []
    labels = [_label(s) for s in summary]
    colors = [SERIES[i % len(SERIES)] for i in range(len(summary))]

    # 1) response latency p50 with p90 whisker
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(summary)), 4))
    p50 = [s["latency_p50_ms"] for s in summary]
    p90 = [s["latency_p90_ms"] for s in summary]
    x = range(len(summary))
    ax.bar(x, p50, width=0.55, color=colors, edgecolor=SURFACE, linewidth=2)
    ax.vlines(x, p50, p90, color=INK2, linewidth=2)
    ax.scatter(x, p90, color=INK2, s=36, zorder=3, label="p90")
    for i, v in enumerate(p50):
        ax.text(i, v / 2, f"{v:,} ms", ha="center", va="center", color="white", fontweight="bold")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("ms from patient's last word to agent audio")
    ax.set_title("Response latency (bar = median, dot = p90) — lower is better", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    paths.append(out / "latency.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    # 2) where the time goes (stacked)
    fig, ax = plt.subplots(figsize=(9, 0.8 * len(summary) + 1.8))
    left = [0.0] * len(summary)
    y = list(range(len(summary)))[::-1]
    for j, (key, name) in enumerate(STAGES):
        vals = [s.get(key) or 0 for s in summary]
        ax.barh(y, vals, left=left, height=0.55, color=SERIES[j], edgecolor=SURFACE, linewidth=2, label=name)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(y, [lb.replace("\n", " ") for lb in labels])
    ax.set_xlabel("average ms per replied turn")
    ax.set_title("Where the latency goes", loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=8)
    fig.tight_layout()
    paths.append(out / "breakdown.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    # 3) conversation-quality side of the trade-off: small multiples, one measure each
    metrics = [("cut_in_rate", "Audible cut-ins (% turns)", True),
               ("backchannel_false_stop_rate", "Stopped on 'mm-hmm' (%)", True),
               ("interruption_stop_p50_ms", "Time to stop on interruption (ms)", False),
               ("tool_accuracy", "Tool routing accuracy (%)*", True)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.6))
    for ax, (key, title, pct) in zip(axes, metrics):
        vals = [(s.get(key) or 0) * (100 if pct else 1) for s in summary]
        ax.bar(x, vals, width=0.6, color=colors, edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", color=INK, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xticks(list(x), [lb.split("\n")[0] for lb in labels], rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "* LLM-routing rows use a simulated LLM that always picks the right tool (best case for the baseline).",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    paths.append(out / "quality.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)
    return paths


def write_report(out: Path, rows: list[dict], ov_rows: list[dict], summary: list[dict], meta: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "turns.csv", rows)
    write_csv(out / "overlaps.csv", ov_rows)
    write_csv(out / "summary.csv", summary)
    imgs = charts(summary, out)
    md = [f"# Benchmark report", "",
          f"- scenarios × variants: {meta['calls']} simulated calls per configuration ({meta['turns']} patient turns)",
          f"- deciders: {', '.join(meta['deciders'])}  ·  decider latency budget: {meta['budget_ms']} ms",
          f"- providers: simulated (see vox/providers/mock.py for latency assumptions)", "",
          markdown(summary), ""]
    for p in imgs:
        md += [f"![{p.stem}]({p.name})", ""]
    md += ["Traces: open any `traces/*.perfetto.json` at https://ui.perfetto.dev", ""]
    (out / "REPORT.md").write_text("\n".join(md))
    return out / "REPORT.md"
