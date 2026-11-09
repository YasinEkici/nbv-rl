# Recorded-path collision audit (D-071)

Descriptive train/VAL geometry measurement only. No policy calls, closed-test input access, trajectory repair, quality-metric or hypothesis changes. Original D-042 nominal geometry and exemptions are retained.

Recorded episodes: 1095; saved IK waypoint gate: passed. Maximum joint sample spacing: 1 degree; near-contact threshold: 10 mm.

Transfers include start/inter-pass movements and return home. Four in-pass subsegments are separately checked and enter the any-collision episode flag; they are excluded from the transfer denominator. Penetration quantiles use the deepest sampled penetration per colliding motion segment, with repeated episode occurrences retained. Primary tail policies/baselines and final-weight train policies remain separate. A safe discrete sample is not a continuous clearance guarantee. Display times linearly map path fractions to recorded time; they do not certify the physical acceleration law.

| Split | Weights | Part | Method | Episodes | Transfers hit / all | Transfer rate | Episodes with collision | Penetration min / median / p95 / max (mm) |
|---|---|---|---|---:|---:|---:|---:|---|
| val | baseline | all | greedy_cost | 15 | 20 / 204 | 9.80% | 15 / 15 (100.00%) | 40.000 / 40.000 / 40.000 / 40.000 |
| val | baseline | all | ilp_tsp | 15 | 0 / 192 | 0.00% | 0 / 15 (0.00%) | none |
| val | baseline | all | joint_opt | 15 | 0 / 193 | 0.00% | 0 / 15 (0.00%) | none |
| val | baseline | all | mcts | 150 | 109 / 1946 | 5.60% | 88 / 150 (58.67%) | 0.174 / 39.999 / 56.836 / 56.836 |
| val | tail | all | myopic/linear | 150 | 212 / 2050 | 10.34% | 150 / 150 (100.00%) | 9.671 / 39.999 / 40.000 / 40.000 |
| val | tail | all | regq/linear | 150 | 216 / 2000 | 10.80% | 120 / 150 (80.00%) | 21.097 / 40.000 / 40.000 / 40.000 |
| val | tail | all | sarsa/convex | 150 | 246 / 2042 | 12.05% | 150 / 150 (100.00%) | 9.671 / 40.000 / 40.000 / 40.000 |
| val | tail | all | sarsa/linear | 150 | 233 / 2050 | 11.37% | 150 / 150 (100.00%) | 9.671 / 40.000 / 40.000 / 40.000 |
| val | tail | all | sarsa/none | 150 | 230 / 2050 | 11.22% | 150 / 150 (100.00%) | 9.671 / 40.000 / 40.000 / 40.000 |
| val | tail | all | sarsa/value_estimate | 150 | 150 / 2036 | 7.37% | 150 / 150 (100.00%) | 13.929 / 40.000 / 40.000 / 40.000 |
| val | baseline | 37275 | greedy_cost | 5 | 5 / 65 | 7.69% | 5 / 5 (100.00%) | 40.000 / 40.000 / 40.000 / 40.000 |
| val | baseline | 37275 | ilp_tsp | 5 | 0 / 65 | 0.00% | 0 / 5 (0.00%) | none |
| val | baseline | 37275 | joint_opt | 5 | 0 / 66 | 0.00% | 0 / 5 (0.00%) | none |
| val | baseline | 37275 | mcts | 50 | 38 / 654 | 5.81% | 32 / 50 (64.00%) | 0.834 / 39.999 / 40.000 / 40.000 |
| val | baseline | 271875 | greedy_cost | 5 | 5 / 73 | 6.85% | 5 / 5 (100.00%) | 40.000 / 40.000 / 40.000 / 40.000 |
| val | baseline | 271875 | ilp_tsp | 5 | 0 / 65 | 0.00% | 0 / 5 (0.00%) | none |
| val | baseline | 271875 | joint_opt | 5 | 0 / 65 | 0.00% | 0 / 5 (0.00%) | none |
| val | baseline | 271875 | mcts | 50 | 38 / 678 | 5.60% | 32 / 50 (64.00%) | 1.739 / 32.952 / 56.836 / 56.836 |
| val | baseline | 591247 | greedy_cost | 5 | 10 / 66 | 15.15% | 5 / 5 (100.00%) | 40.000 / 40.000 / 40.000 / 40.000 |
| val | baseline | 591247 | ilp_tsp | 5 | 0 / 62 | 0.00% | 0 / 5 (0.00%) | none |
| val | baseline | 591247 | joint_opt | 5 | 0 / 62 | 0.00% | 0 / 5 (0.00%) | none |
| val | baseline | 591247 | mcts | 50 | 33 / 614 | 5.37% | 24 / 50 (48.00%) | 0.174 / 40.000 / 40.000 / 40.000 |
| val | tail | 37275 | myopic/linear | 50 | 79 / 650 | 12.15% | 50 / 50 (100.00%) | 24.971 / 40.000 / 40.000 / 40.000 |
| val | tail | 37275 | regq/linear | 50 | 96 / 650 | 14.77% | 50 / 50 (100.00%) | 21.097 / 40.000 / 40.000 / 40.000 |
| val | tail | 37275 | sarsa/convex | 50 | 50 / 650 | 7.69% | 50 / 50 (100.00%) | 40.000 / 40.000 / 40.000 / 40.000 |
| val | tail | 37275 | sarsa/linear | 50 | 50 / 650 | 7.69% | 50 / 50 (100.00%) | 40.000 / 40.000 / 40.000 / 40.000 |
| val | tail | 37275 | sarsa/none | 50 | 50 / 650 | 7.69% | 50 / 50 (100.00%) | 40.000 / 40.000 / 40.000 / 40.000 |
| val | tail | 37275 | sarsa/value_estimate | 50 | 50 / 650 | 7.69% | 50 / 50 (100.00%) | 13.929 / 13.929 / 13.929 / 13.929 |
| val | tail | 271875 | myopic/linear | 50 | 80 / 750 | 10.67% | 50 / 50 (100.00%) | 9.671 / 13.498 / 39.999 / 39.999 |
| val | tail | 271875 | regq/linear | 50 | 20 / 700 | 2.86% | 20 / 50 (40.00%) | 39.983 / 39.983 / 39.983 / 39.983 |
| val | tail | 271875 | sarsa/convex | 50 | 67 / 742 | 9.03% | 50 / 50 (100.00%) | 9.671 / 40.000 / 40.000 / 40.000 |
| val | tail | 271875 | sarsa/linear | 50 | 83 / 750 | 11.07% | 50 / 50 (100.00%) | 9.671 / 40.000 / 40.000 / 40.000 |
| val | tail | 271875 | sarsa/none | 50 | 80 / 750 | 10.67% | 50 / 50 (100.00%) | 9.671 / 40.000 / 40.000 / 40.000 |
| val | tail | 271875 | sarsa/value_estimate | 50 | 50 / 736 | 6.79% | 50 / 50 (100.00%) | 40.000 / 40.000 / 40.000 / 40.000 |
| val | tail | 591247 | myopic/linear | 50 | 53 / 650 | 8.15% | 50 / 50 (100.00%) | 40.000 / 40.000 / 40.000 / 40.000 |
| val | tail | 591247 | regq/linear | 50 | 100 / 650 | 15.38% | 50 / 50 (100.00%) | 40.000 / 40.000 / 40.000 / 40.000 |
| val | tail | 591247 | sarsa/convex | 50 | 129 / 650 | 19.85% | 50 / 50 (100.00%) | 22.706 / 40.000 / 40.000 / 40.000 |
| val | tail | 591247 | sarsa/linear | 50 | 100 / 650 | 15.38% | 50 / 50 (100.00%) | 22.706 / 40.000 / 40.000 / 40.000 |
| val | tail | 591247 | sarsa/none | 50 | 100 / 650 | 15.38% | 50 / 50 (100.00%) | 22.706 / 40.000 / 40.000 / 40.000 |
| val | tail | 591247 | sarsa/value_estimate | 50 | 50 / 650 | 7.69% | 50 / 50 (100.00%) | 40.000 / 40.000 / 40.000 / 40.000 |

## Deepest distinct geometric events

| Part | Variant | Method | Phase | Step | Body | Obstacle | Penetration (mm) | Source |
|---|---|---|---|---:|---|---|---:|---|
| 271875 | 271875_v3 | mcts | transfer | 8 | wrist_3 | part | 56.835735 | p001s6-mcts-n48-val-s7-v4-20261103T170747-fe15a6 |
| 271875 | 271875_v1 | mcts | transfer | 9 | wrist_3 | part | 56.835735 | p001s6-mcts-n48-val-s8-v4-20261103T170747-eda192 |
| 271875 | 271875_v3 | mcts | transfer | 9 | wrist_3 | part | 56.835735 | p001s6-mcts-n48-val-s8-v4-20261103T170747-eda192 |

## Interpretation and scope

Short joint-space paths are not collision-free paths. Per-method rates describe a potential source of comparison bias in the frozen estimated-cycle-time model; they do not establish a new quality ranking or change any reported hypothesis. Compare rates within each part and split rather than pooling different part geometries or final and tail weights. The ledger identifies every affected segment/body/obstacle.

D-021/D-042 checked only saved scan poses, not transfers or the intervals between scan samples. A future ENV v5 proposal is a retract waypoint plus densely checked motion and recomputed time matrices. That requires regenerated baseline and learning results; none of it is applied here.

[Machine-readable event ledger](collision_audit_val_v1.json) · [Original waypoint gate](collision_waypoint_gate_v1.json)

Scope: val. The source corpus may include other scopes; only the listed episodes enter these rates.