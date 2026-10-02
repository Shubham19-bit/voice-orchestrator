# vox — "decide fast, generate slow"

**Latency-optimised orchestration for real-time healthcare voice agents.**
This is a hybrid pipeline: a fast *decision model* ([Jev](https://docs.typesafe.ai) by TypeSafe AI) handles the hot-path decisions, and the LLM only writes the reply.

A cascaded voice agent (speech-to-text → LLM → text-to-speech) spends most of its response time *deciding*, not *talking*:

- **Has the patient finished?** Most stacks wait for a fixed ~700 ms of silence.
- **Which backend call is needed?** Most stacks spend a full LLM round trip just to emit a function call.
- **Was that "mm-hmm" an interruption?** Most stacks stop on any sound.

vox moves those three decisions to a model that returns **typed, calibrated answers in ~100–300 ms**. It runs them in **one fan-out request**, starts the pipeline **speculatively**, and keeps PHI out of every log and trace.

```
mic ─▶ VAD ─┬──────────────────────────────────────────────┐
            └▶ streaming STT ─▶ ORCHESTRATOR ◀─────────────┘
                                  │  one Jev call: {end-of-turn?, which tool?, urgency?}
                                  │  ├─ P(done) high  → reply now
                                  │  ├─ P(done) mid   → start tool+LLM+TTS early, hold audio at a gate
                                  │  └─ P(done) low   → keep listening ("my <pause> um <pause> metformin")
                                  ├─▶ backend tool (timeout + spoken fallback)
                                  ├─▶ LLM stream ─▶ sentence splitter ─▶ TTS stream ─▶ speaker
                                  └─▶ barge-in: backchannel ("mm-hmm") vs real interruption
     every stage → PHI-redacted trace (JSONL + Perfetto timeline)
```

## Results on the bundled simulator

The numbers below come from 6 scripted patient calls × 5 variants, using the offline heuristic decider and simulated providers:

| Strategy | Latency p50 | p90 | Audible cut-ins | Stopped on "mm-hmm" | Interruption stop p50 |
|---|---|---|---|---|---|
| baseline-700 (fixed silence, LLM routing, VAD barge-in) | 1,986 ms | 2,384 ms | 0% | 100% | 340 ms |
| aggressive-400 | 1,674 ms | 2,004 ms | 0% | 100% | 340 ms |
| semantic-eot only | 1,729 ms | 2,163 ms | 0% | 100% | 340 ms |
| **hybrid** | **1,027 ms** | **1,322 ms** | 4% | **0%** | 737 ms |

Each design choice has a cost, and the benchmark shows it:

- Hybrid cuts in on 4% of turns (e.g. "I'm not sure ⏸ when my next…").
- Hybrid takes longer to stop on a real interruption.
- The heuristic decider routes tools less accurately than an (oracle) LLM.

Plugging in real Jev is meant to close those gaps. Measure it; don't assume it.

## Quick start (any laptop, no GPU, no API keys)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m vox.cli demo                       # one call, printed as a timeline
python -m vox.cli demo --scenario elderly-long-pauses --strategy baseline-700
python -m vox.cli bench                      # all strategies → runs/bench-*/REPORT.md + charts
python -m pytest -q                          # 14 tests
```

To see a call as a timeline, open `runs/**/*.perfetto.json` at **https://ui.perfetto.dev**. Each lane is one stage (user, turn, decider, tool, llm, tts, playback, bargein).

The simulator runs in **virtual time**: a 40-second call replays in milliseconds, and the result is deterministic. A real-time run gives the same latencies within about 10 ms (checked).

## mini-Jev: a tiny decision model you train yourself (free, ~4 ms, CPU)

Jev costs money, and a general local LLM (Ollama, qwen2.5:1.5b) measured about **1 s per decision** on a laptop CPU, which is too slow for the hot path. So `vox/minijev/` distils the four decisions this orchestrator needs into a tiny model with **Jev-shaped answers** (noul / choice / score with probabilities and confidence):

```bash
pip install scikit-learn
python -m vox.minijev.train                     # ~15 s: generate data, train 4 heads, evaluate
python -m vox.cli bench --deciders heuristic,minijev
python -m vox.cli live --decider minijev
```

- **Training data** is synthetic: templated patient utterances with fillers, names and disfluencies. End-of-turn negatives come from chopping complete sentences mid-way. The benchmark scenarios are never used for training; every 4+-word benchmark sentence is filtered out.
- **The model** is four heads, each TF-IDF (word + character n-grams, plus explicit "how does it end" tokens) feeding logistic regression.
- **Results:** see `models/minijev_report.md` (accuracy on synthetic and real held-out sets vs. the heuristic, plus latency).

Honest limits: it only answers these four questions, and synthetic data doesn't sound exactly like real patients. On the hand-written scenarios its end-of-turn accuracy is *below* the rule-based heuristic. Those rules were written while looking at the same scenarios, so the heuristic's number is optimistic, but the gap is real. More realistic training data (real transcripts, or labels distilled from a stronger model) is the next step.

## Free, local, Jev-compatible: Ollaya

[Ollaya](https://ollaya.dev) is an open-source local server that speaks Jev's exact `/v1/systemone` format, so the same client code works with no API key and no cost. Your transcripts also never leave your laptop, which is a real privacy advantage for healthcare data.

1. Install it on Windows (PowerShell): `irm https://ollaya.dev/install.ps1 | iex`
2. Start the server: `ollaya serve` (it listens on `127.0.0.1:11435`)
3. Run the project against it:

```bash
python -m vox.cli probe --decider ollaya -n 30
python -m vox.cli bench --deciders heuristic,ollaya
python -m vox.cli live  --decider ollaya
```

The default model is `laya:en`, a small encoder that runs on CPU. If you have an NVIDIA GPU, try `OLLAYA_MODEL=winnow:e4b` in `.env`.

## With Jev (paid, optional)

1. Join the early-access list at typesafe.ai.
2. Once you have a key: `cp .env.example .env` and set `TYPESAFE_API_KEY`.
3. Run these commands:

```bash
python -m vox.cli probe --decider jev -n 30          # measure real Jev latency from your network
python -m vox.cli bench --deciders heuristic,jev     # same calls, real Jev answers + real latencies
python -m vox.cli bench --deciders heuristic,jev,llm # vs. a general LLM in JSON mode (needs LLM_API_KEY)
```

In the benchmark, real decider calls go through `deciders/bridge.py`. It makes the actual HTTP call, measures the true latency, and replays that latency into the simulation. Answers are cached in `runs/cache/`, so re-runs cost nothing. If Jev is slower than `--budget-ms` or returns an error, the turn falls back to the local heuristic, and that fallback is counted.

## Live mode (talk to it)

This needs `DEEPGRAM_API_KEY` (STT + TTS) and `LLM_API_KEY` (Groq's free tier is fine), and optionally `TYPESAFE_API_KEY`. **Use headphones**: there is no echo cancellation yet, so without them the agent hears itself.

```bash
python -m vox.cli live --strategy hybrid --decider jev          # or --decider heuristic-local
python -m vox.cli live --strategy baseline-700                  # feel the difference
```

Each reply prints its latency breakdown, e.g. `⏱ 980 ms after you stopped (wait 340 | route 0 | tool 130 | llm-ttft 290 | tts 210)`.

## Layout

```
vox/orchestrator.py      the core: end-of-turn, speculation gate, pipeline, barge-in, cancellation
vox/config.py            strategies (baseline-700, aggressive-400, semantic-eot, hybrid)
vox/deciders/            jev.py · llm_json.py · heuristic.py · fallback.py (budget + circuit breaker) · bridge.py
vox/tools.py             tool contracts: timeouts + spoken fallbacks
vox/phi.py               redact-at-source + pseudonymous patient ids
vox/tracing.py           per-stage spans → JSONL + Perfetto
vox/sim/                 scripted patients (pauses, STT lag, backchannels, interruptions) + virtual-time runner
vox/eval/                metrics + report/charts
vox/providers/           mock (sim) · Deepgram STT/TTS · OpenAI-compatible LLM · mic/VAD/speaker
```

## Honest limitations

- The simulated provider latencies are assumptions (`providers/mock.py`). Calibrate them against `probe` and live runs.
- The benchmark's LLM routing is an oracle, so the baseline gets best-case tool accuracy.
- The regex PHI redaction demonstrates the principle; it is not a compliance control. Production needs NER (e.g. Presidio), a BAA with every vendor that sees PHI, and review. **Use synthetic data only.**
- The energy VAD is basic. Silero VAD is a drop-in improvement.
- Live mode has no echo cancellation; moving to a browser WebRTC client fixes this.
