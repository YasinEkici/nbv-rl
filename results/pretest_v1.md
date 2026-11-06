# Pretest curve diagnosis and predictions (written before test opening)

| Arm | 200000 steps: mean [seed min, max] | 1000000 steps | 2000000 steps |
|---|---|---|---|
| sarsa/linear/11 | 1.150080 [1.117112, 1.187186] | 1.267094 [1.184591, 1.914797] | 1.207170 [1.206347, 1.209362] |
| regq/linear/11 | 1.181641 [1.176572, 1.187711] | 1.169804 [1.166114, 1.176572] | 1.171129 [1.166753, 1.176572] |
| sarsa/none/11 | 1.218262 [1.195556, 1.241723] | 1.283122 [1.184009, 1.909045] | 1.206435 [1.205921, 1.208079] |
| myopic/linear/11 | 1.219368 [1.206993, 1.238750] | 1.220555 [1.212804, 1.228052] | 1.219536 [1.216867, 1.220509] |
| sarsa/convex/11 | 1.175575 [1.167278, 1.188265] | 1.216711 [1.181212, 1.406387] | 1.198957 [1.191152, 1.205921] |
| sarsa/value_estimate/11 | 1.256258 [1.251537, 1.258188] | 1.131825 [1.128799, 1.135063] | 1.187515 [1.183987, 1.190430] |
| sarsa/linear/16 | 1.223968 [1.214231, 1.239674] | 1.216861 [1.202478, 1.225716] | 1.211862 [1.206917, 1.218059] |

SARSA selected tuning score: 1.162007; official 200000-step mean: 1.150080; official 2000000-step mean: 1.207170.
Dominant descriptive explanation under the predeclared comparison: oscillation. Selection optimism and bounded policy oscillation can coexist; this is not causal identification or a theorem guarantee.
The tuning score averages two tail evaluations; the 200000-step official point uses current weights before its official tail window. Seeds are disjoint and the weight rules differ at those points, limiting direct attribution.
Last-weight versus tail-policy differences support an oscillation interpretation; RegQ is empirically steadier. No extra training or selection.

| Phi | VAL learning-curve area (step × normalized time) | Final VAL minus linear |
|---|---:|---:|
| linear | 2345030.439799 | 0.000000 |
| none | 2389857.477368 | -0.000735 |
| convex | 2325504.372596 | -0.008213 |
| value_estimate | 2231180.231636 | -0.019655 |

Relative AUC range (worst−best)/best: 7.111808%. Trapezoid on observed 100000–2000000 steps (D-055).
Predictions: H1′ rejected; H-forward rejected; H-shaping rejected. Main method remains true online SARSA(lambda).
The pre-tuning small-gap prediction agrees with official VAL; the post-tuning reversal does not. This illustrates selection optimism, without proving it is the sole cause.
RegQ is the best VAL learner and the only learner beating greedy-IG/cost; it remains descriptive, with no hypothesis or primary-method promotion.
