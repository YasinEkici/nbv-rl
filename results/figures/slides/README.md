# Presentation figures (Plan 004b)

Separate presentation exports. Original report figures and the immutable demo are unchanged. All uncertainty/distribution encodings and data splits are stated on the figures and in their JSON receipts.

Presentation principles adapted from [Rougier, Droettboom & Bourne (2014)](https://doi.org/10.1371/journal.pcbi.1003833); the exact layout and typography are project choices.

| Figure | PNG | PDF | Sources and transformations |
|---|---|---|---|
| Measured MCTS / value-estimate decision time: 381× | [PNG](p004b_a_quality_latency.png) | [PDF](p004b_a_quality_latency.pdf) | [JSON](p004b_a_quality_latency.json) |
| Within-run SARSA span 0.408; RegQ span 0.021 | [PNG](p004b_b_learning.png) | [PDF](p004b_b_learning.pdf) | [JSON](p004b_b_learning.json) |
| Test quality varies across parts and individual runs | [PNG](p004b_c_test_parts.png) | [PDF](p004b_c_test_parts.pdf) | [JSON](p004b_c_test_parts.json) |
| Improvement misses the target by 0.000105 | [PNG](p004b_c2_paired_forward.png) | [PDF](p004b_c2_paired_forward.pdf) | [JSON](p004b_c2_paired_forward.json) |
| Joint optimum has the lowest mean VAL movement time | [PNG](p004b_e_val_time_components.png) | [PDF](p004b_e_val_time_components.pdf) | [JSON](p004b_e_val_time_components.json) |
| Unchecked paths collide in 12.0% of SARSA / convex transfers | [PNG](p004b_collision_rates.png) | [PDF](p004b_collision_rates.pdf) | [JSON](p004b_collision_rates.json) |
