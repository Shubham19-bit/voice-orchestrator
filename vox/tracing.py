"""Per-stage latency tracing.

A tiny span recorder (no dependencies) that makes every stage of every turn visible:
turn detection, decider calls, tool calls, LLM time-to-first-token, TTS first byte,
playback. Exports:

  * trace.jsonl         — one span per line (grep/pandas friendly)
  * trace.perfetto.json — Chrome trace format: drag into https://ui.perfetto.dev
                          to see a timeline with one lane per stage.

All string attributes are PHI-redacted *before* they are stored.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .clock import now_ms
from .phi import Redactor

LANES = ["user", "turn", "decider", "tool", "llm", "tts", "playback", "bargein"]


@dataclass
class Span:
    name: str
    lane: str
    start_ms: float
    end_ms: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def dur_ms(self) -> float:
        return (self.end_ms if self.end_ms is not None else self.start_ms) - self.start_ms


class Tracer:
    def __init__(self, redactor: Redactor | None = None, conversation_id: str = "conv"):
        self.redactor = redactor or Redactor()
        self.conversation_id = conversation_id
        self.spans: list[Span] = []

    def _clean(self, attrs: dict[str, Any]) -> dict[str, Any]:
        return {k: (self.redactor.redact(v) if isinstance(v, str) else v) for k, v in attrs.items()}

    def start(self, name: str, lane: str, **attrs) -> Span:
        sp = Span(name, lane, now_ms(), attrs=self._clean(attrs))
        self.spans.append(sp)
        return sp

    def end(self, sp: Span, **attrs) -> None:
        sp.end_ms = now_ms()
        sp.attrs.update(self._clean(attrs))

    @contextmanager
    def span(self, name: str, lane: str, **attrs):
        sp = self.start(name, lane, **attrs)
        try:
            yield sp
        except BaseException as e:  # includes CancelledError
            sp.attrs["status"] = type(e).__name__
            raise
        finally:
            if sp.end_ms is None:
                sp.end_ms = now_ms()

    def event(self, name: str, lane: str, **attrs) -> None:
        t = now_ms()
        self.spans.append(Span(name, lane, t, t, self._clean(attrs)))

    # ---------------------------------------------------------------- export
    def export(self, out_dir: Path, prefix: str = "trace") -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{prefix}.jsonl", "w") as f:
            for s in self.spans:
                f.write(json.dumps({"conv": self.conversation_id, "name": s.name, "lane": s.lane,
                                    "start_ms": round(s.start_ms, 2), "dur_ms": round(s.dur_ms, 2),
                                    **s.attrs}) + "\n")
        events = []
        for i, lane in enumerate(LANES):
            events.append({"ph": "M", "name": "thread_name", "pid": 1, "tid": i, "args": {"name": lane}})
        for s in self.spans:
            tid = LANES.index(s.lane) if s.lane in LANES else len(LANES)
            ev = {"name": s.name, "pid": 1, "tid": tid, "ts": s.start_ms * 1000, "args": s.attrs}
            if s.dur_ms > 0:
                ev.update(ph="X", dur=s.dur_ms * 1000)
            else:
                ev.update(ph="i", s="t")
            events.append(ev)
        with open(out_dir / f"{prefix}.perfetto.json", "w") as f:
            json.dump({"traceEvents": events, "displayTimeUnit": "ms",
                       "metadata": {"exported_at": time.time()}}, f)
