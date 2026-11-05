# Plan 002 tuning grid (exploratory)

VAL score = mean of the final two evaluations, using running tail-mean weights; lower is better.
Spread is population standard deviation across seeds 1000–1002. No test-part evaluation.

## sarsa

| point | setting | seed 1000 | seed 1001 | seed 1002 | mean | spread | success rate | diverged |
|---|---|---|---|---|---|---|---|---|
| sarsa-00 | {"alpha": 0.01, "epsilon": "linear", "lam": 0.5} | 1.292145 | 1.285746 | 1.295464 | 1.291118 | 0.004033 | 1.0000 | 0/3 |
| sarsa-01 | {"alpha": 0.01, "epsilon": "constant", "lam": 0.5} | 1.250500 | 1.254744 | 1.262080 | 1.255774 | 0.004783 | 1.0000 | 0/3 |
| sarsa-02 | {"alpha": 0.01, "epsilon": "linear", "lam": 0.9} | 1.289370 | 1.282383 | 1.321457 | 1.297737 | 0.017014 | 1.0000 | 0/3 |
| sarsa-03 | {"alpha": 0.01, "epsilon": "constant", "lam": 0.9} | 1.278241 | 1.278881 | 1.291317 | 1.282813 | 0.006019 | 1.0000 | 0/3 |
| sarsa-04 | {"alpha": 0.03, "epsilon": "linear", "lam": 0.5} | 1.198340 | 1.198755 | 1.199759 | 1.198951 | 0.000596 | 1.0000 | 0/3 |
| sarsa-05 | {"alpha": 0.03, "epsilon": "constant", "lam": 0.5} | 1.171045 | 1.166146 | 1.148831 | 1.162007 | 0.009529 | 1.0000 | 0/3 |
| sarsa-06 | {"alpha": 0.03, "epsilon": "linear", "lam": 0.9} | 1.187434 | 1.194504 | 1.202014 | 1.194651 | 0.005953 | 1.0000 | 0/3 |
| sarsa-07 | {"alpha": 0.03, "epsilon": "constant", "lam": 0.9} | 1.558150 | 1.201377 | 1.197784 | 1.319104 | 0.169038 | 0.8333 | 0/3 |
| sarsa-08 | {"alpha": 0.1, "epsilon": "linear", "lam": 0.5} | 1.185202 | 1.186807 | 1.197566 | 1.189858 | 0.005489 | 1.0000 | 0/3 |
| sarsa-09 | {"alpha": 0.1, "epsilon": "constant", "lam": 0.5} | 1.189095 | 1.192232 | 1.183690 | 1.188339 | 0.003528 | 1.0000 | 0/3 |
| sarsa-10 | {"alpha": 0.1, "epsilon": "linear", "lam": 0.9} | 1.172310 | 1.192294 | 1.186634 | 1.183746 | 0.008410 | 1.0000 | 0/3 |
| sarsa-11 | {"alpha": 0.1, "epsilon": "constant", "lam": 0.9} | 1.190338 | 1.186999 | 1.197070 | 1.191469 | 0.004189 | 1.0000 | 0/3 |

Selected: sarsa-05, {"alpha": 0.03, "epsilon": "constant", "lam": 0.5}.
Rule: within 0.005 of the minimum mean, lowest seed spread, then smallest alpha.

## regq

| point | setting | seed 1000 | seed 1001 | seed 1002 | mean | spread | success rate | diverged |
|---|---|---|---|---|---|---|---|---|
| regq-00 | {"alpha": 0.01, "epsilon": "linear", "eta": 0.001} | 1.192243 | 1.192243 | 1.192243 | 1.192243 | 0.000000 | 1.0000 | 0/3 |
| regq-01 | {"alpha": 0.01, "epsilon": "constant", "eta": 0.001} | 1.182029 | 1.181798 | 1.181798 | 1.181875 | 0.000109 | 1.0000 | 0/3 |
| regq-02 | {"alpha": 0.01, "epsilon": "linear", "eta": 0.01} | 1.191377 | 1.191377 | 1.191377 | 1.191377 | 0.000000 | 1.0000 | 0/3 |
| regq-03 | {"alpha": 0.01, "epsilon": "constant", "eta": 0.01} | 1.200502 | 1.200502 | 1.200502 | 1.200502 | 0.000000 | 1.0000 | 0/3 |
| regq-04 | {"alpha": 0.03, "epsilon": "linear", "eta": 0.001} | 1.192243 | 1.192243 | 1.191935 | 1.192141 | 0.000145 | 1.0000 | 0/3 |
| regq-05 | {"alpha": 0.03, "epsilon": "constant", "eta": 0.001} | 1.178789 | 1.179763 | 1.179408 | 1.179320 | 0.000402 | 1.0000 | 0/3 |
| regq-06 | {"alpha": 0.03, "epsilon": "linear", "eta": 0.01} | 1.191377 | 1.191377 | 1.191377 | 1.191377 | 0.000000 | 1.0000 | 0/3 |
| regq-07 | {"alpha": 0.03, "epsilon": "constant", "eta": 0.01} | 1.200445 | 1.200502 | 1.200502 | 1.200483 | 0.000027 | 1.0000 | 0/3 |
| regq-08 | {"alpha": 0.1, "epsilon": "linear", "eta": 0.001} | 1.192600 | 1.192243 | 1.192243 | 1.192362 | 0.000168 | 1.0000 | 0/3 |
| regq-09 | {"alpha": 0.1, "epsilon": "constant", "eta": 0.001} | 1.191627 | 1.184337 | 1.184307 | 1.186757 | 0.003444 | 1.0000 | 0/3 |
| regq-10 | {"alpha": 0.1, "epsilon": "linear", "eta": 0.01} | 1.191377 | 1.191377 | 1.191377 | 1.191377 | 0.000000 | 1.0000 | 0/3 |
| regq-11 | {"alpha": 0.1, "epsilon": "constant", "eta": 0.01} | 1.200445 | 1.203076 | 1.200502 | 1.201341 | 0.001227 | 1.0000 | 0/3 |

Selected: regq-01, {"alpha": 0.01, "epsilon": "constant", "eta": 0.001}.
Rule: within 0.005 of the minimum mean, lowest seed spread, then smallest alpha.

## myopic

| point | setting | seed 1000 | seed 1001 | seed 1002 | mean | spread | success rate | diverged |
|---|---|---|---|---|---|---|---|---|
| myopic-00 | {"alpha": 0.01, "epsilon": "linear", "phi_alpha": 25.0} | 1.231878 | 1.231800 | 1.229937 | 1.231205 | 0.000897 | 1.0000 | 0/3 |
| myopic-01 | {"alpha": 0.01, "epsilon": "constant", "phi_alpha": 25.0} | 1.226987 | 1.228873 | 1.227471 | 1.227777 | 0.000800 | 1.0000 | 0/3 |
| myopic-02 | {"alpha": 0.01, "epsilon": "linear", "phi_alpha": 50.0} | 1.222752 | 1.229937 | 1.230150 | 1.227613 | 0.003438 | 1.0000 | 0/3 |
| myopic-03 | {"alpha": 0.01, "epsilon": "constant", "phi_alpha": 50.0} | 1.229745 | 1.228335 | 1.226444 | 1.228175 | 0.001353 | 1.0000 | 0/3 |
| myopic-04 | {"alpha": 0.03, "epsilon": "linear", "phi_alpha": 25.0} | 1.426090 | 1.259360 | 1.614018 | 1.433156 | 0.144874 | 1.0000 | 0/3 |
| myopic-05 | {"alpha": 0.03, "epsilon": "constant", "phi_alpha": 25.0} | 1.288667 | 1.226776 | 1.487059 | 1.334167 | 0.111024 | 1.0000 | 0/3 |
| myopic-06 | {"alpha": 0.03, "epsilon": "linear", "phi_alpha": 50.0} | 1.220200 | 1.221625 | 1.226146 | 1.222657 | 0.002535 | 1.0000 | 0/3 |
| myopic-07 | {"alpha": 0.03, "epsilon": "constant", "phi_alpha": 50.0} | 1.226292 | 1.222910 | 1.223390 | 1.224197 | 0.001494 | 1.0000 | 0/3 |
| myopic-08 | {"alpha": 0.1, "epsilon": "linear", "phi_alpha": 25.0} | 1.840721 | 1.957940 | 1.905630 | 1.901430 | 0.047946 | 0.3333 | 0/3 |
| myopic-09 | {"alpha": 0.1, "epsilon": "constant", "phi_alpha": 25.0} | 1.903073 | 1.879303 | 1.878454 | 1.886944 | 0.011411 | 0.4889 | 0/3 |
| myopic-10 | {"alpha": 0.1, "epsilon": "linear", "phi_alpha": 50.0} | 1.633568 | 1.584161 | 1.561468 | 1.593065 | 0.030101 | 0.6889 | 0/3 |
| myopic-11 | {"alpha": 0.1, "epsilon": "constant", "phi_alpha": 50.0} | 1.728070 | 1.881164 | 1.868692 | 1.825975 | 0.069416 | 0.6667 | 0/3 |

Selected: myopic-07, {"alpha": 0.03, "epsilon": "constant", "phi_alpha": 50.0}.
Rule: within 0.005 of the minimum mean, lowest seed spread, then smallest alpha.

## Selected settings and canonical VAL references

| method | VAL selection score | seed spread | success rate |
|---|---|---|---|
| sarsa | 1.162007 | 0.009529 | 1.0000 |
| regq | 1.181875 | 0.000109 | 1.0000 |
| myopic | 1.224197 | 0.001494 | 1.0000 |
| greedy_cost (reference) | 1.186 | — | 1.0000 |
| mcts (reference) | 1.104 | — | 1.0000 |
| ilp_tsp (reference) | 1.000 | — | 1.0000 |
