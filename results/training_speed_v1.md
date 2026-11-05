# Plan 002 single-core smoke measurements

10000 training steps per row, 20 VAL evaluations. Learning steps/s excludes evaluation time;
nominal model/feature initialization inside the learning loop is included. All rows are exploratory.

| method | none | linear | convex | value_estimate (cached) | value_estimate (uncached) | cache speedup |
|---|---|---|---|---|---|---|
| sarsa | 1061.0 | 971.2 | 1100.8 | 1144.4 | 765.3 | 1.495x |
| regq | 1695.7 | 1619.3 | 1625.2 | 1240.9 | 845.2 | 1.468x |
| myopic | 1832.9 | 1780.2 | 1735.6 | 1239.6 | 852.5 | 1.454x |

16-feature linear SARSA: 1562.8 learning steps/s.

Estimated official wall time on 20 cores: **1.551 hours**.
Includes 30 main seeds, 30 SARSA Phi-ablation seeds and 10 exploratory full-feature seeds (all 2000000 steps).
Formula: sum over 70 seeds of (200 × smoke learning seconds + smoke evaluation seconds), divided by 20.
This is an ideal parallel estimate; thermal throttling and contention can increase wall time.
