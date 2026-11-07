# Frozen-policy train evaluation

No training or selection. All 60 canonical checkpoints, both weight kinds; D-041 train baselines reused.

| Arm | Part | Normalized cycle time | Raw seconds | Success |
|---|---|---:|---:|---:|
| sarsa/linear | 124336 | 1.142786 | 67.221145 | 1.000000 |
| sarsa/linear | 124532 | 1.148787 | 52.820904 | 1.000000 |
| sarsa/linear | 135365 | 1.206259 | 74.194875 | 1.000000 |
| sarsa/linear | 46262 | 1.370151 | 51.809226 | 1.000000 |
| sarsa/linear | 548863 | 1.275685 | 45.353638 | 1.000000 |
| sarsa/linear | 635751 | 1.197680 | 46.837622 | 1.000000 |
| sarsa/linear | 65149 | 1.005825 | 58.696760 | 1.000000 |
| sarsa/linear | 702389 | 1.225968 | 37.462119 | 1.000000 |
| regq/linear | 124336 | 1.118691 | 65.793018 | 1.000000 |
| regq/linear | 124532 | 1.154649 | 53.090465 | 1.000000 |
| regq/linear | 135365 | 1.118280 | 68.754343 | 1.000000 |
| regq/linear | 46262 | 1.204843 | 45.558475 | 1.000000 |
| regq/linear | 548863 | 1.238990 | 44.082257 | 1.000000 |
| regq/linear | 635751 | 1.165747 | 45.594805 | 1.000000 |
| regq/linear | 65149 | 1.001297 | 58.416950 | 1.000000 |
| regq/linear | 702389 | 1.222460 | 37.359294 | 1.000000 |
| myopic/linear | 124336 | 1.159146 | 68.185779 | 1.000000 |
| myopic/linear | 124532 | 1.197777 | 55.073489 | 1.000000 |
| myopic/linear | 135365 | 1.190863 | 73.249167 | 1.000000 |
| myopic/linear | 46262 | 1.408201 | 53.248019 | 1.000000 |
| myopic/linear | 548863 | 1.372372 | 48.784742 | 1.000000 |
| myopic/linear | 635751 | 1.244869 | 48.669800 | 1.000000 |
| myopic/linear | 65149 | 1.006161 | 58.715530 | 1.000000 |
| myopic/linear | 702389 | 1.239845 | 38.115564 | 1.000000 |
| sarsa/none | 124336 | 1.145349 | 67.376429 | 1.000000 |
| sarsa/none | 124532 | 1.148607 | 52.812640 | 1.000000 |
| sarsa/none | 135365 | 1.205674 | 74.157209 | 1.000000 |
| sarsa/none | 46262 | 1.369979 | 51.802717 | 1.000000 |
| sarsa/none | 548863 | 1.273715 | 45.284911 | 1.000000 |
| sarsa/none | 635751 | 1.197680 | 46.837622 | 1.000000 |
| sarsa/none | 65149 | 1.006161 | 58.715530 | 1.000000 |
| sarsa/none | 702389 | 1.224930 | 37.431703 | 1.000000 |
| sarsa/convex | 124336 | 1.149508 | 67.622859 | 1.000000 |
| sarsa/convex | 124532 | 1.147853 | 52.777996 | 1.000000 |
| sarsa/convex | 135365 | 1.200083 | 73.813546 | 1.000000 |
| sarsa/convex | 46262 | 1.378368 | 52.119958 | 1.000000 |
| sarsa/convex | 548863 | 1.299561 | 46.200138 | 1.000000 |
| sarsa/convex | 635751 | 1.186469 | 46.375152 | 1.000000 |
| sarsa/convex | 65149 | 1.002129 | 58.469556 | 1.000000 |
| sarsa/convex | 702389 | 1.225126 | 37.438266 | 1.000000 |
| sarsa/value_estimate | 124336 | 1.141201 | 67.134473 | 1.000000 |
| sarsa/value_estimate | 124532 | 1.096086 | 50.397753 | 1.000000 |
| sarsa/value_estimate | 135365 | 1.125722 | 69.244673 | 1.000000 |
| sarsa/value_estimate | 46262 | 1.193214 | 45.118748 | 1.000000 |
| sarsa/value_estimate | 548863 | 1.387403 | 49.301594 | 1.000000 |
| sarsa/value_estimate | 635751 | 1.107675 | 43.306022 | 1.000000 |
| sarsa/value_estimate | 65149 | 1.003614 | 58.500139 | 1.000000 |
| sarsa/value_estimate | 702389 | 1.241680 | 37.931176 | 1.000000 |
| random | 124336 | 1.944531 | 114.326277 | 0.500000 |
| random | 124532 | 2.579974 | 118.626513 | 0.320000 |
| random | 135365 | 1.864956 | 114.631700 | 0.200000 |
| random | 46262 | 2.825051 | 106.823078 | 0.500000 |
| random | 548863 | 2.502751 | 88.818491 | 0.300000 |
| random | 635751 | 2.600744 | 101.576693 | 0.580000 |
| random | 65149 | 1.977978 | 115.000919 | 0.280000 |
| random | 702389 | 3.018705 | 92.305851 | 0.920000 |
| fixed | 124336 | 1.358588 | 79.890164 | 1.000000 |
| fixed | 124532 | 1.627718 | 74.842042 | 1.000000 |
| fixed | 135365 | 1.246600 | 76.659396 | 1.000000 |
| fixed | 46262 | 1.784872 | 67.490995 | 1.000000 |
| fixed | 548863 | 1.559942 | 55.395697 | 1.000000 |
| fixed | 635751 | 1.567121 | 61.332442 | 1.000000 |
| fixed | 65149 | 1.428998 | 83.043964 | 1.000000 |
| fixed | 702389 | 2.026050 | 61.920503 | 1.000000 |
| greedy_ig | 124336 | 1.141143 | 67.112408 | 1.000000 |
| greedy_ig | 124532 | 1.195925 | 54.988336 | 1.000000 |
| greedy_ig | 135365 | 1.200706 | 73.843477 | 1.000000 |
| greedy_ig | 46262 | 1.368981 | 51.764998 | 1.000000 |
| greedy_ig | 548863 | 1.294398 | 46.022714 | 1.000000 |
| greedy_ig | 635751 | 1.220754 | 47.755856 | 1.000000 |
| greedy_ig | 65149 | 1.017497 | 59.376025 | 1.000000 |
| greedy_ig | 702389 | 1.227782 | 37.515293 | 1.000000 |
| greedy_cost | 124336 | 1.119154 | 65.841224 | 1.000000 |
| greedy_cost | 124532 | 1.149894 | 52.871833 | 1.000000 |
| greedy_cost | 135365 | 1.125722 | 69.244673 | 1.000000 |
| greedy_cost | 46262 | 1.200991 | 45.412811 | 1.000000 |
| greedy_cost | 548863 | 1.383668 | 49.170676 | 1.000000 |
| greedy_cost | 635751 | 1.174751 | 45.929002 | 1.000000 |
| greedy_cost | 65149 | 1.016570 | 59.266811 | 1.000000 |
| greedy_cost | 702389 | 1.273011 | 39.043145 | 1.000000 |
| ilp_tsp | 124336 | 1.000000 | 58.837871 | 1.000000 |
| ilp_tsp | 124532 | 1.000000 | 45.979737 | 1.000000 |
| ilp_tsp | 135365 | 1.000000 | 61.542053 | 1.000000 |
| ilp_tsp | 46262 | 1.000000 | 37.812795 | 1.000000 |
| ilp_tsp | 548863 | 1.000000 | 35.558991 | 1.000000 |
| ilp_tsp | 635751 | 1.000000 | 39.252757 | 1.000000 |
| ilp_tsp | 65149 | 1.000000 | 58.375378 | 1.000000 |
| ilp_tsp | 702389 | 1.000000 | 30.818499 | 1.000000 |
| mcts | 124336 | 1.036375 | 60.959216 | 1.000000 |
| mcts | 124532 | 1.098379 | 50.503198 | 1.000000 |
| mcts | 135365 | 1.102263 | 67.788233 | 1.000000 |
| mcts | 46262 | 1.155741 | 43.701792 | 1.000000 |
| mcts | 548863 | 1.190013 | 42.271064 | 1.000000 |
| mcts | 635751 | 1.128183 | 44.165122 | 1.000000 |
| mcts | 65149 | 1.001516 | 58.300649 | 1.000000 |
| mcts | 702389 | 1.041587 | 31.930049 | 1.000000 |
| joint_opt | 124336 | 0.964760 | 56.755855 | 1.000000 |
| joint_opt | 124532 | 0.986035 | 45.337629 | 1.000000 |
| joint_opt | 135365 | 1.050574 | 64.646001 | 1.000000 |
| joint_opt | 46262 | 0.986158 | 37.289404 | 1.000000 |
| joint_opt | 548863 | 0.981530 | 34.903113 | 1.000000 |
| joint_opt | 635751 | 1.000000 | 39.252757 | 1.000000 |
| joint_opt | 65149 | 0.953239 | 55.537234 | 1.000000 |
| joint_opt | 702389 | 1.000000 | 30.818499 | 1.000000 |

Split means (equal part/variant/seed weights):

| Arm | Part | Normalized cycle time | Raw seconds | Success |
|---|---|---:|---:|---:|
| sarsa/linear | split | 1.196642 | 54.299536 | 1.000000 |
| regq/linear | split | 1.153120 | 52.331201 | 1.000000 |
| myopic/linear | split | 1.227404 | 55.505261 | 1.000000 |
| sarsa/none | split | 1.196512 | 54.302345 | 1.000000 |
| sarsa/convex | split | 1.198637 | 54.352184 | 1.000000 |
| sarsa/value_estimate | split | 1.162074 | 52.616822 | 1.000000 |
| random | split | 2.414336 | 106.513690 | 0.450000 |
| fixed | split | 1.574986 | 70.071900 | 1.000000 |
| greedy_ig | split | 1.208398 | 54.797388 | 1.000000 |
| greedy_cost | split | 1.180470 | 53.347522 | 1.000000 |
| ilp_tsp | split | 1.000000 | 46.022260 | 1.000000 |
| mcts | split | 1.094257 | 49.952415 | 1.000000 |
| joint_opt | split | 0.990287 | 45.567561 | 1.000000 |

Secondary final weights:

| Arm | Part | Normalized cycle time | Raw seconds | Success |
|---|---|---:|---:|---:|
| sarsa/linear | split | 1.236684 | 55.908460 | 1.000000 |
| regq/linear | split | 1.143678 | 52.039280 | 1.000000 |
| myopic/linear | split | 1.236354 | 55.835217 | 1.000000 |
| sarsa/none | split | 1.347643 | 60.616263 | 0.927500 |
| sarsa/convex | split | 1.240327 | 56.028585 | 1.000000 |
| sarsa/value_estimate | split | 1.163128 | 52.715026 | 1.000000 |
| random | split | 2.414336 | 106.513690 | 0.450000 |
| fixed | split | 1.574986 | 70.071900 | 1.000000 |
| greedy_ig | split | 1.208398 | 54.797388 | 1.000000 |
| greedy_cost | split | 1.180470 | 53.347522 | 1.000000 |
| ilp_tsp | split | 1.000000 | 46.022260 | 1.000000 |
| mcts | split | 1.094257 | 49.952415 | 1.000000 |
| joint_opt | split | 0.990287 | 45.567561 | 1.000000 |

## Model limitations (reported with every result)

- Ideal sensor (D-025): a sample is measured or not (binary); in reality precision falls gradually with the incidence angle (first order ~1/cos α). Multiple reflections on shiny surfaces, the lower accuracy at the ends of the laser line, and measurement noise are not modeled; the camera's field of view is assumed to cover the whole line. The analytic sweep itself only neglects the discrete profile spacing (0.1 mm at 100 mm/s and 1 kHz).
- Single setup without turntable (D-025, D-027): interior and table-facing surfaces are not measurable by any optical sensor from outside; coverage is relative to the reachable union (D-023).
- Reachability is checked at 5 samples along each sweep with a joint-step limit (D-021); transfers between passes are not checked. Robot–part and robot–table collisions are checked statically at those samples with capsules and a sensor-head box (D-042); transfers, self-collision and fixtures are not.
- Cycle time excludes data processing (registration, CAD comparison) (D-036).
