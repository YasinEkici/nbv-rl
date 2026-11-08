# Candidate-mask audit (A3)

Only the eleven train/VAL base parts were read; the three test caches were not opened. A differing allowed pair rules out identical masks across all fourteen; individual test masks are not inspected.

| Part | Split | Reachable | IK failure | Collision masked |
|---|---|---:|---:|---:|
| 271875 | val | 34 | 9 | 5 |
| 46262 | train | 34 | 9 | 5 |
| 124336 | train | 34 | 9 | 5 |
| 548863 | train | 37 | 8 | 3 |
| 37275 | val | 34 | 9 | 5 |
| 591247 | val | 34 | 9 | 5 |
| 65149 | train | 33 | 10 | 5 |
| 702389 | train | 33 | 8 | 7 |
| 135365 | train | 34 | 9 | 5 |
| 124532 | train | 33 | 12 | 3 |
| 635751 | train | 34 | 9 | 5 |

Identical allowed mask_reason arrays: **False**. Identical directions: **True**. Identical sensor start/end positions: **False**.

Directions are part-independent Fibonacci-cap vectors (candidates.py). Center, support/standoff and sweep length depend on the nominal mesh. robot.plan_passes solves each part separately: IK/table/step constraints, then min(robot-part, robot-table) clearance at five samples. No constant mask is hard-coded. Equal masks are an empirical coincidence in this fixed cell/normalized size, not proof that part geometry is irrelevant. Combined cached min_clearance alone does not identify the limiting body or establish table-only causation.

Source path and raw SHA-256 per allowed cache, mask arrays and geometry hashes: [JSON](candidate_audit_v1.json). No new IK, ray casting or evaluation.

## Existing first-VAL clearance evidence

The following rounded diagnostics were already recorded in results/collision_v2.csv. Only its first VAL-part prefix was read; no later test rows. These are supporting diagnostics, not a fresh collision computation.

| Pass | Part clearance mm / body | Table clearance mm / body |
|---:|---|---|
| 5 | -31.12 / upper_arm | 16.22 / sensor_head |
| 8 | 9.87 / upper_arm | -41.2 / sensor_head |
| 13 | -5.17 / upper_arm | -61.7 / sensor_head |
| 21 | 36.24 / upper_arm | -19.91 / sensor_head |
| 33 | 87.4 / forearm | -21.58 / upper_arm |

Both part and table violations occur in this saved example; do not attribute every masked direction to the table. IK failure does not identify a single cause either. For the other parts the combined cache alone is insufficient to separate causes; we are uncertain rather than claiming a causal diagnostic.

Allowed counterexample (zero-based pass indices): `{"conclusion": "This allowed pair alone disproves identical mask_reason arrays across the full fourteen-part set.", "differing_pass_indices": [9, 20, 21, 33, 43], "parts": [271875, 548863]}`.
