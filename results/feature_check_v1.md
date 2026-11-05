## Plan 002 exploratory feature check (D-046)

VAL normalized cycle time; mean of evaluations at 190000 and 200000 steps using the running tail mean.

| seed | 16 features | 11 features | improvement (11 minus 16) |
|---|---|---|---|
| 1000 | 1.261260 | 1.187434 | -0.073826 |
| 1001 | 1.333844 | 1.194504 | -0.139340 |
| 1002 | 1.213517 | 1.202014 | -0.011503 |
| Mean | 1.269540 | 1.194651 | -0.074890 |

D-046 rule (improvement >= 0.01): select 11; drop features 9 and 13–16.

Final tail-policy success rate, 16 features: 1.0000.
Final tail-policy success rate, 11 features: 1.0000.
Canonical val reference, greedy_cost: 1.186.
Canonical val reference, mcts: 1.104.

These exploratory results select features only; they are not official method comparisons.
