# Recorded-path collision audit (D-071)

Descriptive train/VAL geometry measurement only. No policy calls, closed-test input access, trajectory repair, quality-metric or hypothesis changes. Original D-042 nominal geometry and exemptions are retained.

Recorded episodes: 4495; saved IK waypoint gate: passed. Maximum joint sample spacing: 1 degree; near-contact threshold: 10 mm.

Transfers include start/inter-pass movements and return home. Four in-pass subsegments are separately checked and enter the any-collision episode flag; they are excluded from the transfer denominator. Penetration quantiles use the deepest sampled penetration per colliding motion segment, with repeated episode occurrences retained. Primary tail policies/baselines and final-weight train policies remain separate. A safe discrete sample is not a continuous clearance guarantee. Display times linearly map path fractions to recorded time; they do not certify the physical acceleration law.

| Split | Weights | Part | Method | Episodes | Transfers hit / all | Transfer rate | Episodes with collision | Penetration min / median / p95 / max (mm) |
|---|---|---|---|---:|---:|---:|---:|---|
| train | baseline | all | fixed | 40 | 50 / 630 | 7.94% | 35 / 40 (87.50%) | 4.641 / 32.618 / 45.905 / 45.905 |
| train | baseline | all | greedy_cost | 40 | 32 / 470 | 6.81% | 26 / 40 (65.00%) | 13.367 / 37.511 / 40.000 / 40.000 |
| train | baseline | all | greedy_ig | 40 | 47 / 465 | 10.11% | 31 / 40 (77.50%) | 3.254 / 36.364 / 40.000 / 40.000 |
| train | baseline | all | ilp_tsp | 40 | 1 / 450 | 0.22% | 1 / 40 (2.50%) | 37.511 / 37.511 / 37.511 / 37.511 |
| train | baseline | all | joint_opt | 40 | 1 / 449 | 0.22% | 1 / 40 (2.50%) | 37.511 / 37.511 / 37.511 / 37.511 |
| train | baseline | all | mcts | 400 | 353 / 4518 | 7.81% | 284 / 400 (71.00%) | 0.005 / 26.456 / 40.000 / 49.613 |
| train | baseline | all | random | 400 | 765 / 9403 | 8.14% | 348 / 400 (87.00%) | 0.005 / 27.367 / 40.000 / 64.291 |
| train | tail | all | myopic/linear | 400 | 599 / 4745 | 12.62% | 360 / 400 (90.00%) | 3.254 / 29.845 / 40.162 / 41.623 |
| train | tail | all | regq/linear | 400 | 361 / 4569 | 7.90% | 194 / 400 (48.50%) | 0.011 / 30.438 / 40.000 / 40.000 |
| train | tail | all | sarsa/convex | 400 | 415 / 4688 | 8.85% | 253 / 400 (63.25%) | 3.254 / 25.700 / 40.000 / 40.000 |
| train | tail | all | sarsa/linear | 400 | 450 / 4660 | 9.66% | 250 / 400 (62.50%) | 3.254 / 25.700 / 40.000 / 40.000 |
| train | tail | all | sarsa/none | 400 | 444 / 4663 | 9.52% | 250 / 400 (62.50%) | 3.254 / 25.700 / 40.000 / 40.000 |
| train | tail | all | sarsa/value_estimate | 400 | 303 / 4768 | 6.35% | 246 / 400 (61.50%) | 5.463 / 35.356 / 40.000 / 40.000 |
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
| train | baseline | 46262 | fixed | 5 | 5 / 75 | 6.67% | 5 / 5 (100.00%) | 4.641 / 4.641 / 4.641 / 4.641 |
| train | baseline | 46262 | greedy_cost | 5 | 1 / 50 | 2.00% | 1 / 5 (20.00%) | 40.000 / 40.000 / 40.000 / 40.000 |
| train | baseline | 46262 | greedy_ig | 5 | 6 / 55 | 10.91% | 5 / 5 (100.00%) | 11.014 / 40.000 / 40.000 / 40.000 |
| train | baseline | 46262 | ilp_tsp | 5 | 0 / 45 | 0.00% | 0 / 5 (0.00%) | none |
| train | baseline | 46262 | joint_opt | 5 | 0 / 45 | 0.00% | 0 / 5 (0.00%) | none |
| train | baseline | 46262 | mcts | 50 | 24 / 488 | 4.92% | 24 / 50 (48.00%) | 9.928 / 40.000 / 40.000 / 40.000 |
| train | baseline | 46262 | random | 50 | 97 / 1182 | 8.21% | 41 / 50 (82.00%) | 0.335 / 26.865 / 40.000 / 40.000 |
| train | baseline | 65149 | fixed | 5 | 5 / 90 | 5.56% | 5 / 5 (100.00%) | 45.905 / 45.905 / 45.905 / 45.905 |
| train | baseline | 65149 | greedy_cost | 5 | 8 / 63 | 12.70% | 5 / 5 (100.00%) | 26.456 / 40.000 / 40.000 / 40.000 |
| train | baseline | 65149 | greedy_ig | 5 | 5 / 62 | 8.06% | 5 / 5 (100.00%) | 40.000 / 40.000 / 40.000 / 40.000 |
| train | baseline | 65149 | ilp_tsp | 5 | 0 / 67 | 0.00% | 0 / 5 (0.00%) | none |
| train | baseline | 65149 | joint_opt | 5 | 0 / 66 | 0.00% | 0 / 5 (0.00%) | none |
| train | baseline | 65149 | mcts | 50 | 53 / 634 | 8.36% | 39 / 50 (78.00%) | 0.005 / 26.157 / 45.905 / 45.905 |
| train | baseline | 65149 | random | 50 | 84 / 1220 | 6.89% | 44 / 50 (88.00%) | 0.005 / 24.659 / 40.000 / 45.905 |
| train | baseline | 124336 | fixed | 5 | 10 / 85 | 11.76% | 5 / 5 (100.00%) | 18.151 / 29.075 / 40.000 / 40.000 |
| train | baseline | 124336 | greedy_cost | 5 | 5 / 71 | 7.04% | 5 / 5 (100.00%) | 40.000 / 40.000 / 40.000 / 40.000 |
| train | baseline | 124336 | greedy_ig | 5 | 12 / 69 | 17.39% | 5 / 5 (100.00%) | 3.254 / 15.866 / 40.000 / 40.000 |
| train | baseline | 124336 | ilp_tsp | 5 | 0 / 70 | 0.00% | 0 / 5 (0.00%) | none |
| train | baseline | 124336 | joint_opt | 5 | 0 / 69 | 0.00% | 0 / 5 (0.00%) | none |
| train | baseline | 124336 | mcts | 50 | 47 / 670 | 7.01% | 35 / 50 (70.00%) | 4.154 / 13.011 / 40.000 / 40.000 |
| train | baseline | 124336 | random | 50 | 105 / 1216 | 8.63% | 45 / 50 (90.00%) | 0.011 / 25.024 / 40.000 / 40.000 |
| train | baseline | 124532 | fixed | 5 | 5 / 80 | 6.25% | 5 / 5 (100.00%) | 39.998 / 39.998 / 39.998 / 39.998 |
| train | baseline | 124532 | greedy_cost | 5 | 5 / 55 | 9.09% | 5 / 5 (100.00%) | 30.438 / 30.438 / 30.438 / 30.438 |
| train | baseline | 124532 | greedy_ig | 5 | 10 / 55 | 18.18% | 5 / 5 (100.00%) | 30.438 / 33.413 / 36.388 / 36.388 |
| train | baseline | 124532 | ilp_tsp | 5 | 0 / 55 | 0.00% | 0 / 5 (0.00%) | none |
| train | baseline | 124532 | joint_opt | 5 | 0 / 55 | 0.00% | 0 / 5 (0.00%) | none |
| train | baseline | 124532 | mcts | 50 | 64 / 550 | 11.64% | 46 / 50 (92.00%) | 5.799 / 30.438 / 39.993 / 39.993 |
| train | baseline | 124532 | random | 50 | 85 / 1218 | 6.98% | 44 / 50 (88.00%) | 2.560 / 20.175 / 40.000 / 40.000 |
| train | baseline | 135365 | fixed | 5 | 5 / 86 | 5.81% | 5 / 5 (100.00%) | 5.689 / 5.689 / 5.689 / 5.689 |
| train | baseline | 135365 | greedy_cost | 5 | 0 / 77 | 0.00% | 0 / 5 (0.00%) | none |
| train | baseline | 135365 | greedy_ig | 5 | 8 / 78 | 10.26% | 5 / 5 (100.00%) | 7.018 / 7.018 / 35.848 / 35.848 |
| train | baseline | 135365 | ilp_tsp | 5 | 0 / 78 | 0.00% | 0 / 5 (0.00%) | none |
| train | baseline | 135365 | joint_opt | 5 | 0 / 79 | 0.00% | 0 / 5 (0.00%) | none |
| train | baseline | 135365 | mcts | 50 | 24 / 773 | 3.10% | 20 / 50 (40.00%) | 0.447 / 9.460 / 39.999 / 40.000 |
| train | baseline | 135365 | random | 50 | 123 / 1234 | 9.97% | 48 / 50 (96.00%) | 0.099 / 27.709 / 40.000 / 44.289 |
| train | baseline | 548863 | fixed | 5 | 0 / 75 | 0.00% | 0 / 5 (0.00%) | none |
| train | baseline | 548863 | greedy_cost | 5 | 5 / 63 | 7.94% | 5 / 5 (100.00%) | 36.364 / 40.000 / 40.000 / 40.000 |
| train | baseline | 548863 | greedy_ig | 5 | 5 / 56 | 8.93% | 5 / 5 (100.00%) | 36.364 / 36.364 / 36.364 / 36.364 |
| train | baseline | 548863 | ilp_tsp | 5 | 0 / 51 | 0.00% | 0 / 5 (0.00%) | none |
| train | baseline | 548863 | joint_opt | 5 | 0 / 51 | 0.00% | 0 / 5 (0.00%) | none |
| train | baseline | 548863 | mcts | 50 | 33 / 546 | 6.04% | 32 / 50 (64.00%) | 1.607 / 36.364 / 40.000 / 40.000 |
| train | baseline | 548863 | random | 50 | 117 / 1194 | 9.80% | 44 / 50 (88.00%) | 1.607 / 37.604 / 45.361 / 64.291 |
| train | baseline | 635751 | fixed | 5 | 10 / 69 | 14.49% | 5 / 5 (100.00%) | 21.638 / 30.819 / 40.000 / 40.000 |
| train | baseline | 635751 | greedy_cost | 5 | 0 / 50 | 0.00% | 0 / 5 (0.00%) | none |
| train | baseline | 635751 | greedy_ig | 5 | 1 / 50 | 2.00% | 1 / 5 (20.00%) | 26.946 / 26.946 / 26.946 / 26.946 |
| train | baseline | 635751 | ilp_tsp | 5 | 0 / 48 | 0.00% | 0 / 5 (0.00%) | none |
| train | baseline | 635751 | joint_opt | 5 | 0 / 48 | 0.00% | 0 / 5 (0.00%) | none |
| train | baseline | 635751 | mcts | 50 | 59 / 497 | 11.87% | 42 / 50 (84.00%) | 5.463 / 26.946 / 39.999 / 40.000 |
| train | baseline | 635751 | random | 50 | 98 / 1123 | 8.73% | 44 / 50 (88.00%) | 2.388 / 31.346 / 40.000 / 49.728 |
| train | baseline | 702389 | fixed | 5 | 10 / 70 | 14.29% | 5 / 5 (100.00%) | 25.982 / 32.618 / 39.253 / 39.253 |
| train | baseline | 702389 | greedy_cost | 5 | 8 / 41 | 19.51% | 5 / 5 (100.00%) | 13.367 / 13.367 / 37.511 / 37.511 |
| train | baseline | 702389 | greedy_ig | 5 | 0 / 40 | 0.00% | 0 / 5 (0.00%) | none |
| train | baseline | 702389 | ilp_tsp | 5 | 1 / 36 | 2.78% | 1 / 5 (20.00%) | 37.511 / 37.511 / 37.511 / 37.511 |
| train | baseline | 702389 | joint_opt | 5 | 1 / 36 | 2.78% | 1 / 5 (20.00%) | 37.511 / 37.511 / 37.511 / 37.511 |
| train | baseline | 702389 | mcts | 50 | 49 / 360 | 13.61% | 46 / 50 (92.00%) | 11.056 / 13.699 / 29.031 / 49.613 |
| train | baseline | 702389 | random | 50 | 56 / 1016 | 5.51% | 38 / 50 (76.00%) | 0.419 / 15.584 / 40.000 / 40.685 |
| train | tail | 46262 | myopic/linear | 50 | 72 / 555 | 12.97% | 50 / 50 (100.00%) | 5.655 / 23.875 / 40.000 / 40.000 |
| train | tail | 46262 | regq/linear | 50 | 0 / 500 | 0.00% | 0 / 50 (0.00%) | none |
| train | tail | 46262 | sarsa/convex | 50 | 31 / 550 | 5.64% | 30 / 50 (60.00%) | 11.014 / 40.000 / 40.000 / 40.000 |
| train | tail | 46262 | sarsa/linear | 50 | 60 / 550 | 10.91% | 50 / 50 (100.00%) | 11.014 / 32.850 / 40.000 / 40.000 |
| train | tail | 46262 | sarsa/none | 50 | 60 / 550 | 10.91% | 50 / 50 (100.00%) | 11.014 / 32.850 / 40.000 / 40.000 |
| train | tail | 46262 | sarsa/value_estimate | 50 | 50 / 500 | 10.00% | 50 / 50 (100.00%) | 40.000 / 40.000 / 40.000 / 40.000 |
| train | tail | 65149 | myopic/linear | 50 | 50 / 620 | 8.06% | 50 / 50 (100.00%) | 40.000 / 40.000 / 40.000 / 40.000 |
| train | tail | 65149 | regq/linear | 50 | 70 / 620 | 11.29% | 50 / 50 (100.00%) | 26.456 / 40.000 / 40.000 / 40.000 |
| train | tail | 65149 | sarsa/convex | 50 | 54 / 621 | 8.70% | 50 / 50 (100.00%) | 26.456 / 40.000 / 40.000 / 40.000 |
| train | tail | 65149 | sarsa/linear | 50 | 50 / 620 | 8.06% | 50 / 50 (100.00%) | 40.000 / 40.000 / 40.000 / 40.000 |
| train | tail | 65149 | sarsa/none | 50 | 50 / 620 | 8.06% | 50 / 50 (100.00%) | 40.000 / 40.000 / 40.000 / 40.000 |
| train | tail | 65149 | sarsa/value_estimate | 50 | 57 / 628 | 9.08% | 50 / 50 (100.00%) | 26.456 / 40.000 / 40.000 / 40.000 |
| train | tail | 124336 | myopic/linear | 50 | 150 / 710 | 21.13% | 50 / 50 (100.00%) | 3.254 / 15.866 / 40.000 / 40.000 |
| train | tail | 124336 | regq/linear | 50 | 188 / 691 | 27.21% | 50 / 50 (100.00%) | 0.011 / 1.855 / 40.000 / 40.000 |
| train | tail | 124336 | sarsa/convex | 50 | 200 / 706 | 28.33% | 50 / 50 (100.00%) | 3.254 / 15.866 / 40.000 / 40.000 |
| train | tail | 124336 | sarsa/linear | 50 | 200 / 690 | 28.99% | 50 / 50 (100.00%) | 3.254 / 15.577 / 40.000 / 40.000 |
| train | tail | 124336 | sarsa/none | 50 | 200 / 693 | 28.86% | 50 / 50 (100.00%) | 3.254 / 15.866 / 40.000 / 40.000 |
| train | tail | 124336 | sarsa/value_estimate | 50 | 46 / 710 | 6.48% | 46 / 50 (92.00%) | 40.000 / 40.000 / 40.000 / 40.000 |
| train | tail | 124532 | myopic/linear | 50 | 107 / 550 | 19.45% | 50 / 50 (100.00%) | 20.219 / 30.438 / 39.993 / 39.993 |
| train | tail | 124532 | regq/linear | 50 | 59 / 550 | 10.73% | 50 / 50 (100.00%) | 25.700 / 30.438 / 30.438 / 30.438 |
| train | tail | 124532 | sarsa/convex | 50 | 55 / 550 | 10.00% | 50 / 50 (100.00%) | 25.700 / 30.438 / 30.438 / 30.438 |
| train | tail | 124532 | sarsa/linear | 50 | 90 / 550 | 16.36% | 50 / 50 (100.00%) | 25.700 / 30.438 / 30.438 / 30.438 |
| train | tail | 124532 | sarsa/none | 50 | 84 / 550 | 15.27% | 50 / 50 (100.00%) | 25.700 / 30.438 / 30.438 / 30.438 |
| train | tail | 124532 | sarsa/value_estimate | 50 | 50 / 550 | 9.09% | 50 / 50 (100.00%) | 30.438 / 30.438 / 30.438 / 30.438 |
| train | tail | 135365 | myopic/linear | 50 | 60 / 790 | 7.59% | 50 / 50 (100.00%) | 7.018 / 7.018 / 40.000 / 40.000 |
| train | tail | 135365 | regq/linear | 50 | 10 / 748 | 1.34% | 10 / 50 (20.00%) | 7.588 / 7.588 / 7.588 / 7.588 |
| train | tail | 135365 | sarsa/convex | 50 | 50 / 790 | 6.33% | 50 / 50 (100.00%) | 7.018 / 7.018 / 7.018 / 7.018 |
| train | tail | 135365 | sarsa/linear | 50 | 50 / 790 | 6.33% | 50 / 50 (100.00%) | 7.018 / 7.018 / 7.018 / 7.018 |
| train | tail | 135365 | sarsa/none | 50 | 50 / 790 | 6.33% | 50 / 50 (100.00%) | 7.018 / 7.018 / 7.018 / 7.018 |
| train | tail | 135365 | sarsa/value_estimate | 50 | 0 / 770 | 0.00% | 0 / 50 (0.00%) | none |
| train | tail | 548863 | myopic/linear | 50 | 100 / 610 | 16.39% | 50 / 50 (100.00%) | 10.754 / 40.000 / 41.623 / 41.623 |
| train | tail | 548863 | regq/linear | 50 | 34 / 560 | 6.07% | 34 / 50 (68.00%) | 10.754 / 40.000 / 40.000 / 40.000 |
| train | tail | 548863 | sarsa/convex | 50 | 25 / 571 | 4.38% | 23 / 50 (46.00%) | 40.000 / 40.000 / 40.000 / 40.000 |
| train | tail | 548863 | sarsa/linear | 50 | 0 / 560 | 0.00% | 0 / 50 (0.00%) | none |
| train | tail | 548863 | sarsa/none | 50 | 0 / 560 | 0.00% | 0 / 50 (0.00%) | none |
| train | tail | 548863 | sarsa/value_estimate | 50 | 0 / 660 | 0.00% | 0 / 50 (0.00%) | none |
| train | tail | 635751 | myopic/linear | 50 | 50 / 500 | 10.00% | 50 / 50 (100.00%) | 15.133 / 15.133 / 15.133 / 15.133 |
| train | tail | 635751 | regq/linear | 50 | 0 / 500 | 0.00% | 0 / 50 (0.00%) | none |
| train | tail | 635751 | sarsa/convex | 50 | 0 / 500 | 0.00% | 0 / 50 (0.00%) | none |
| train | tail | 635751 | sarsa/linear | 50 | 0 / 500 | 0.00% | 0 / 50 (0.00%) | none |
| train | tail | 635751 | sarsa/none | 50 | 0 / 500 | 0.00% | 0 / 50 (0.00%) | none |
| train | tail | 635751 | sarsa/value_estimate | 50 | 100 / 500 | 20.00% | 50 / 50 (100.00%) | 5.463 / 20.409 / 35.356 / 35.356 |
| train | tail | 702389 | myopic/linear | 50 | 10 / 410 | 2.44% | 10 / 50 (20.00%) | 15.809 / 15.809 / 15.809 / 15.809 |
| train | tail | 702389 | regq/linear | 50 | 0 / 400 | 0.00% | 0 / 50 (0.00%) | none |
| train | tail | 702389 | sarsa/convex | 50 | 0 / 400 | 0.00% | 0 / 50 (0.00%) | none |
| train | tail | 702389 | sarsa/linear | 50 | 0 / 400 | 0.00% | 0 / 50 (0.00%) | none |
| train | tail | 702389 | sarsa/none | 50 | 0 / 400 | 0.00% | 0 / 50 (0.00%) | none |
| train | tail | 702389 | sarsa/value_estimate | 50 | 0 / 450 | 0.00% | 0 / 50 (0.00%) | none |
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
| 548863 | 548863_v4 | random | transfer | 8 | sensor_head | part | 64.291237 | p001s6-random-n48-train-s9-v4-20261103T170747-9439c1 |
| 271875 | 271875_v3 | mcts | transfer | 8 | wrist_3 | part | 56.835735 | p001s6-mcts-n48-val-s7-v4-20261103T170747-fe15a6 |
| 271875 | 271875_v1 | mcts | transfer | 9 | wrist_3 | part | 56.835735 | p001s6-mcts-n48-val-s8-v4-20261103T170747-eda192 |

## Interpretation and scope

Short joint-space paths are not collision-free paths. Per-method rates describe a potential source of comparison bias in the frozen estimated-cycle-time model; they do not establish a new quality ranking or change any reported hypothesis. Compare rates within each part and split rather than pooling different part geometries or final and tail weights. The ledger identifies every affected segment/body/obstacle.

D-021/D-042 checked only saved scan poses, not transfers or the intervals between scan samples. A future ENV v5 proposal is a retract waypoint plus densely checked motion and recomputed time matrices. That requires regenerated baseline and learning results; none of it is applied here.

[Machine-readable event ledger](collision_audit_v1.json) · [Original waypoint gate](collision_waypoint_gate_v1.json)

train: ILP + TSP has 0.22% colliding transfers versus MCTS 7.81%; ILP's rate is higher on 0 of 8 parts. This is a descriptive comparison of recorded paths, not a causal relation between joint-path length and collision. Different pass counts change the transfer denominator. Feasible collision-aware cycle times were not measured, so neither the magnitude nor direction of a corrected quality-ranking bias is established.
val: ILP + TSP has 0.00% colliding transfers versus MCTS 5.60%; ILP's rate is higher on 0 of 3 parts. This is a descriptive comparison of recorded paths, not a causal relation between joint-path length and collision. Different pass counts change the transfer denominator. Feasible collision-aware cycle times were not measured, so neither the magnitude nor direction of a corrected quality-ranking bias is established.

## Interpretation (observation, not a decision)

`robot.movej_time` uses d = max_k |dq_k| and the monotone rest-to-rest trapezoidal/triangular duration: d/v + v/a when d >= v^2/a, otherwise 2*sqrt(d/a). `robot.transfer` passes the actual limit-checked, multi-turn endpoint displacement to that function; ENV v4 uses `robot.movej_time` for return home as well. For the same unwrapped endpoints, zero endpoint velocities and the same joint velocity/acceleration limits, a straight joint path minimizes each joint's travel and achieves the largest-axis time bound. A collision-free detour cannot have a shorter duration under these assumptions. Intermediate stops can only add constraints. This is a lower bound for the fixed endpoints, not a collision-free trajectory construction or an optimization over alternative scan poses, winding choices or pass sequences.

Consequently the recorded time of a colliding transfer is an optimistic lower bound on its collision-free replacement time. The sampled transfer rates below include return home and are generated from the measured event ledger.

| Split | Method | Colliding transfers |
|---|---|---:|
| train | greedy_cost | 6.81% |
| train | ilp_tsp | 0.22% |
| train | joint_opt | 0.22% |
| train | mcts | 7.81% |
| train | myopic/linear | 12.62% |
| train | regq/linear | 7.90% |
| train | sarsa/convex | 8.85% |
| train | sarsa/linear | 9.66% |
| train | sarsa/none | 9.52% |
| train | sarsa/value_estimate | 6.35% |
| val | greedy_cost | 9.80% |
| val | ilp_tsp | 0.00% |
| val | joint_opt | 0.00% |
| val | mcts | 5.60% |
| val | myopic/linear | 10.34% |
| val | regq/linear | 10.80% |
| val | sarsa/convex | 12.05% |
| val | sarsa/linear | 11.37% |
| val | sarsa/none | 11.22% |
| val | sarsa/value_estimate | 7.37% |

If the ILP + TSP reference remains feasible at its recorded cost, adding nonnegative avoidance time to a competing fixed sequence can only increase that sequence's normalized cycle time. The observed near-zero CAD-planner rates therefore support the Phase 1 finding that the main linear learner does not match the CAD reference. However, ILP + TSP itself has sampled train collisions, so its normalization denominator may also increase; zero sampled collisions is not a continuous safety certificate. The fixed-endpoint bound does not prove every hypothesis robust or determine corrected normalized rankings.

val: SARSA/value_estimate has 7.37% colliding transfers; MCTS has 5.60%.
train: SARSA/value_estimate has 6.35% colliding transfers; MCTS has 7.81%.
Their collision-rate ordering reverses between VAL and train.
The reported test-quality proximity of SARSA/value_estimate and MCTS remains within this unquantified modeling uncertainty. Additional collision-free time was not measured; a collision frequency alone does not determine its size.

No test-part collision measurement was performed: the holdout is closed and independent test joint-step traces are not available from the registry. Any statement about test robustness here is an analogy from VAL/train, not a test-set geometry finding. No result, setting, policy choice or hypothesis verdict is changed.

Scope: all. The source corpus may include other scopes; only the listed episodes enter these rates.
Scope status: {"val": "complete", "train-primary": "complete", "train-final": "not measured"}. Unmeasured episodes never enter rates.