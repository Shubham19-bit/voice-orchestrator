# mini-Jev training report

- training examples: eot=17,286, tool=5,100, urgency=5,100, overlap=3,000
- training time: 3.4 s (CPU)
- inference latency (eot + tool + urgency together): p50 5.36 ms, p95 7.86 ms

| Decision | mini-Jev (synthetic test) | mini-Jev (real scenarios) | Heuristic (real scenarios) | n real |
|---|---|---|---|---|
| End of turn (noul) | 95% | **83%** | 85% | 47 |
| Tool routing (choice) | 100% | **96%** | 89% | 27 |
| Urgency (score) | 100% | **100%** | 96% | 27 |
| Backchannel vs interruption (choice) | 100% | **100%** | 100% | 13 |

Real scenarios are the hand-written benchmark calls (never used for training). They are small,
so treat differences of one or two examples with care.