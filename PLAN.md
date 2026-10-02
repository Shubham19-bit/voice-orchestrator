# Build plan: 14 days to an interview-ready project

**Goal:** walk into the 100ms interview with a working system and one honest, measured claim. For example: *"Moving hot-path decisions to Jev cut median response latency from ~2.0 s to ~1.0 s on 150 simulated patient turns, at the cost of X% cut-ins, which I then reduced to Y%."*

**Already built:** the simulator, orchestrator, heuristic decider, Jev client, benchmark, report and tests all run today. Your job is to understand every line, add real providers, add real Jev, and push the numbers.

## Day 0 (today, ~1 hour)

- [ ] Join the Jev early-access waitlist (typesafe.ai). This is the only thing that can block you.
- [ ] Sign up for Deepgram (free credits) and Groq (free tier). Put the keys in `.env`.
- [ ] `pip install -r requirements.txt`, then run `python -m vox.cli demo`, `python -m vox.cli bench` and `python -m pytest -q`.
- [ ] Push to GitHub (private for now). Commit daily: a commit history is evidence of real work.

## Week 1: own the system

| Day | Do | You should be able to explain |
|---|---|---|
| 1 | Read `orchestrator.py` end to end. Run `demo` for every scenario and strategy, and open the traces in Perfetto. | Why `baseline-700` can never be faster than 700 ms plus the pipeline. |
| 2 | Read `clock.py`, `sim/player.py` and `eval/metrics.py`. Change a scenario's pauses and predict the metric change *before* running it. | How virtual time works (selector patch), and why STT lag matters at the first end-of-turn check. |
| 3 | **Live mode.** `python -m vox.cli live --strategy baseline-700`, then `hybrid --decider heuristic-local`. Record a 60-second screen capture. | Real stage latencies vs. the mock profile. Update `MockProfile` with your measured numbers. |
| 4 | `probe` the LLM decider (`--decider llm`). If Jev access has arrived, run `probe --decider jev -n 50`. | p50/p90 decision latency, and where cold TLS connections show up. |
| 5 | **Jev in the benchmark:** `bench --deciders heuristic,jev,llm`. Tune `commit_prob`, `speculate_prob` and `hold_ms` on a *held-out* scenario split. Don't tune on the test set. | The precision/latency trade-off of each threshold, and why you pin `JEV_MODEL` after tuning. |
| 6 | Write 6 more scenarios (accents, a noisy line, a caregiver speaking for the patient, numbers/dates). | Where heuristic and Jev disagree, with examples. |
| 7 | Buffer day and write-up draft #1: problem, design, results table, three failure cases. | |

## Week 2: improve the trade-off, then polish

Pick 2–3 of these. Each is a self-contained, measurable improvement:

1. **Reduce cut-ins.** Use the urgency/"sounds incomplete" signal to raise `hold_ms` only for slow speakers. Or re-check P(done) right before opening the speculation gate.
2. **Faster barge-in without false stops.** *Duck* the agent's volume immediately on speech onset, then fully stop or resume once the decider answers. This cuts perceived stop latency from ~740 ms to ~VAD time.
3. **Speculative tool prefetch.** Start the backend call at the *first* mid-confidence decision, before commit. Measure the wasted calls vs. the latency saved.
4. **Escalation path.** Urgency ≥ 1.5 bypasses the LLM: immediate scripted safety line plus a nurse callback. Measure urgent recall.
5. **Better PHI redaction.** Add Presidio NER and a unit test with 30 synthetic PHI strings. Report leak rate.
6. **OpenTelemetry export** of the same spans to Jaeger (`docker run jaegertracing/all-in-one`). This shows production-style observability.
7. **Browser client over WebRTC** (aiortc or LiveKit) for echo cancellation and real network jitter.

| Day | Do |
|---|---|
| 8–11 | Two improvements from the list, each with a before/after row in the benchmark table. |
| 12 | Final bench run with the frozen config. Charts go into README. Record a 2-minute demo video (live call + Perfetto trace). |
| 13 | Two-page technical write-up (it mirrors the "technical design doc" deliverable in the PS-II description). |
| 14 | Mock interview: answer the questions below out loud, without notes. |

## How this maps to the 100ms project description

| They wrote | Where it is |
|---|---|
| end-to-end latency budgeting & profiling | `tracing.py`, the Response timestamps, `breakdown.png` |
| streaming & partial-result handling | re-deciding on every STT update; sentence-level TTS streaming |
| turn-taking & barge-in detection | `_silence_watch`, `_handle_overlap` |
| graceful recovery when a model or hop degrades | `FallbackDecider` (budget + circuit breaker), tool timeouts + spoken fallbacks |
| tool-calling contracts | `tools.py` |
| PHI out of logs/traces, still debuggable | `phi.py` redact-at-source, pseudonymous ids, test that traces contain no names |
| measure before optimising | baseline first; every change is a benchmark row |
| instrumentation harness (expected outcome 2) | `bench` + Perfetto traces |

## Questions to be ready for

- Why asyncio and not threads? What happens to an in-flight HTTP stream when you `task.cancel()`?
- How do you stop speech *instantly* on barge-in when audio is already buffered in the device or the network? (Buffer sizing, flush, ducking.)
- WebRTC vs. WebSocket for audio: UDP vs. TCP, jitter buffers, head-of-line blocking, echo cancellation.
- Your p90 is 30% above p50. Where does tail latency come from? (Cold connections, LLM queueing, TTS first byte.)
- What if Jev is down? Walk through the circuit breaker.
- Why does the simulator use virtual time, and how do you know it matches reality? (Real-time parity check.)
- What would you need before sending real patient transcripts to a third-party API? (BAA, minimum necessary, retention, audit.)
- Speculative execution wastes calls. What's the cost model? (Calls per turn × price vs. ms saved.)
