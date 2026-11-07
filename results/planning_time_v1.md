# Deployed decision latency (exploratory)

| Policy | Source split | Mean ms/decision | Median ms | P95 ms | Calls |
|---|---|---:|---:|---:|---:|
| greedy_cost | val | 0.516845 | 0.508500 | 0.556640 | 567 |
| mcts | val | 49.265584 | 44.271900 | 112.356240 | 567 |
| regq/linear | val | 0.127334 | 0.125000 | 0.155770 | 567 |
| sarsa/linear | val | 0.128641 | 0.124300 | 0.155680 | 567 |
| sarsa/value_estimate | val | 0.129405 | 0.126400 | 0.155980 | 567 |

Serial warm-cache inference calls on the SAME greedy-cost VAL states; feature evaluation/argmax or nominal MCTS included; environment steps, model preprocessing, cache loading and Phi computation excluded. Phi is training-only for frozen policies. Fixed official seed 0 checkpoints, no quality selection. Source checkpoints are fixed official seed 0; timings characterize this machine and state sample, not a seed-level quality comparison.
