# Descriptive cross-split finding

| SARSA Phi | Train | VAL | Test |
|---|---:|---:|---:|
| linear | 1.196642 | 1.207170 | 1.252614 |
| none | 1.196512 | 1.206435 | 1.244123 |
| convex | 1.198637 | 1.198957 | 1.227824 |
| value_estimate | 1.162074 | 1.187515 | 1.142396 |
| MCTS reference | 1.094257 | 1.104443 | 1.144764 |

H-forward paired SARSA-minus-myopic difference: -0.019894820622543; missing improvement to the predeclared 0.02 threshold: 0.000105179377457. No rounding-based reinterpretation.

value_estimate is the lowest SARSA mean on each split. This is descriptive, outside the hypotheses; no arm or setting is selected from test results. Cross-split consistency is supporting evidence, not an independent replication. Six tested arms expose a test-only best-arm claim to selection optimism.
