# Problem definition, preregistration and evaluation protocol

The experiment protocol was frozen on **2026-11-03**; the classical-learning design was approved on **2026-11-04**. Public editorial consolidation: **2026-11-27**. No experiment definition, setting, hypothesis, threshold or result changes. Historical design/prediction dates are retained. The completed test holdout is closed.

## Problem and assignment

Graduate Deep Reinforcement Learning, Homework 7: plan next views for a robot-mounted line-laser profilometer so a cast/metal part reaches 95% measurable coverage with minimum estimated robot-motion plus scan time, generalizing to unseen part types. The assignment suggests continuous sensor-direction actions, information-gain rewards and SAC. The completed midterm study uses discrete candidate passes, classical NumPy learning with linear features, joint-space motion time, and learner-side potential shaping. Deep methods require separate approval and preregistration.

Nominal CAD is known. Pass geometry, IK, nominal visibility and approximate planning matrices come from CAD. The real variant determines measurements but its geometry/visibility are privileged. The observation contains covered-sample bits, used passes, current joints, step count and nominal coverage; the nominal model is available. Policies cannot inspect the real variant or evaluation-only fields.

## Environment definition

- 48 passes: Fibonacci directions on the upper cap, elevation at least 10 degrees; standoff is nominal directional support plus 150 mm. Sweep is global x projected perpendicular to view, with length equal to part projection plus 20 mm at each end. Used, IK-unreachable and statically colliding passes are masked.
- 20000 area-weighted samples per base part, with correspondence across variants. A measured sample crosses the swept laser plane, lies within the 150–250 mm depth interval and laser fan, satisfies laser incidence/camera angle limits of 65 degrees, and is unoccluded to both camera and laser. Both part and table occlude. Camera-only visibility is a secondary ablation.
- UR5 base (−350, 0, 0) mm, yaw 180 degrees; sensor origin 120 mm along flange z; joint limits ±360 degrees; home (0, −90, 90, −90, −90, 0) degrees. Five IK samples per pass; consecutive per-joint changes at most 45 degrees; frame/sensor origins at least 20 mm above table. Capsules and sensor box require 5 mm clearance at these samples. Base, shoulder housing and first 65 mm of upper arm retain the table exemption.
- Leading-joint triangular/trapezoidal move time uses speed 1.05 rad/s and acceleration 1.4 rad/s². Transfers choose the nearest equivalent joint start unless the whole shifted sweep violates limits, then the closest feasible equivalent sweep. Scan time is sweep length / 100 mm/s. Planning matrices do not replace executed joint/time accounting.
- Terminate at 95% real reachable-union coverage, 24 passes, empty legal mask, or a policy/repair returning no action. All endings are terminal and charge home return.
- Reward is −(move + scan) per pass, −home at termination, plus one 250 s success bonus. Undiscounted return is −cycle time + 250 × success. Potential shaping is outside the environment objective.

## Data and split

14 Thingi10K mechanical source models proxy cast parts: 8 train, 3 validation, 3 test, stratified by hard-to-see share. Split is by base part; all five variants remain together. Longest side is 200 mm. Deviation seed 0 produces low-frequency displacement up to 0.5 mm, dents/bumps up to 1 mm and per-axis scale within ±0.75%. This is not a measured casting dataset. Author/source/license-family attribution is retained; missing license versions must not be inferred.

Only train is used for learning; all feature/parameter/shaping selections use validation. Test parts never guide adaptive selection.

## Metrics

Primary coverage uses the real reachable union under the 48 candidates. Secondary coverage uses the common 128-candidate union; raw coverage uses all samples. No method runs with 128 candidates in this experiment.

Primary time includes moves, scans, repair and home. Normalized cycle time is executed time divided by the executed **same-variant, same-candidate-set ILP + TSP time**. Raw seconds and success accompany every ratio. ILP + TSP is a reference, not proof of physical global optimality.

Below-target endings are censored: elapsed time plus home remains in the mean, with failure retained in success. Secondary metrics: passes to target; coverage at 5/10 passes (final coverage if earlier terminated); normalized coverage/time area on [0,180 s] using a step function with jumps at completed passes; total absolute joint travel including from/to home; repair passes; success without repair; train minus test gaps in normalized time, raw seconds and success. Home adds time, not coverage.

## Comparators

Main references: cost-aware greedy information gain; nominal-model UCT MCTS; minimum-cardinality set-cover ILP followed by TSP; joint CP-SAT pass selection/order. The joint planner is a nominal-CAD quality reference, not a real-visibility oracle. Plan-ahead methods repair with nominal cost-aware greedy when their stored plan ends below target. Auxiliary references: random, fixed top/ring pattern and gain-only greedy. Real-visibility greedy is diagnostic only.

Stochastic references use seeds 0–9, deterministic references one run. Original solver settings, deterministic configuration, hints, 300 s joint-solver limit and bound recording remain unchanged.

## Preregistered methods and features

Primary method: **true online SARSA(λ) with linear potential**, chosen before results. Comparisons: regularized linear Q-learning and a myopic one-step learner. SARSA potential ablation: none, linear, convex, estimated remaining value. Stronger later results never change the primary method.

Discount is 1; learner rewards and potentials are scaled by 0.01. Potentials: 0; 50C; 50C²; minus estimated remaining nominal greedy-cost scan/motion/home time. C is measured share of the nominal reachable union. All terminal potentials are zero. Myopic regression uses the one-step shaped reward without bootstrap.

The 11 ordered features are bias, gain, gain_rel, t_move, t_scan, rate_rel, finishes, t_home, remaining, needed, step_frac. Features use observation and nominal model only. Time features divide seconds by 10. State-only components add the same scalar to all actions. Linear-potential shaping is equivalent to a bias/remaining weight initialization in this representation; this was checked at multiple λ values, not asserted for arbitrary potentials.

The preregistered removal control compares 16 versus 11 features, training on train and evaluating validation. Five extras remain only if they improve normalized time by at least 0.01. Settings: seeds 1000–1002; 200000 steps; step 0.03 / feature count; λ 0.9; linear exploration schedule; linear potential. The 11-feature decision preceded tuning.

## Hypotheses

### H1′: close half the greedy-to-planner gap

For test part p, g_p is cost-aware greedy mean normalized time, x_p primary SARSA mean. Threshold: (1 + g_p)/2. Both the split mean criterion and at least 2 of 3 part thresholds must hold. If not met, report each (g_p − x_p)/(g_p − 1), retaining negative values, and position relative to MCTS.

### H-forward: contribution beyond a myopic target

Primary SARSA must improve mean normalized time over myopic by at least 0.02. Pair matching variants after separately averaging method seeds. SARSA minus myopic must also be negative on at least 2 of 3 parts. Otherwise report actual paired differences; threshold failure does not prove equivalent policies.

### H-shaping: learning speed changes, final performance stays close

Speed uses validation normalized time integrated over the common observed training-step grid, without an invented step-zero score. Require (worst area − best area)/best area ≥ 10%. Final closeness requires every potential test mean within 0.02 of the linear mean. Both must hold. Otherwise report which potential helps/hurts and by how much. None versus linear is an initialization comparison. Objective invariance does not guarantee faster learning.

RegQ has no confirmatory hypothesis: report it descriptively with divergence statistics. Camera-only coverage overestimate is also descriptive.

## Budgets, tuning and selection

- Tuning seeds **1000–1002**, official seeds **0–9**. Every learner gets **12 settings × 3 seeds × 200000 steps**.
- SARSA grid: step {0.01,0.03,0.1} / feature count; λ {0.5,0.9}; either linear exploration 0.2→0.01 during the first half or constant 0.05.
- RegQ: direct step {0.01,0.03,0.1}; regularization {0.001,0.01}; same exploration schedules.
- Myopic: direct step {0.01,0.03,0.1}; potential scale {25,50} s; same schedules.
- Uniform train-variant sampling by episode with matched environment-seed sequences across methods; learner and environment seeds are separate.
- Greedy validation (no exploration) uses all 15 variants. Tuning has 20 evaluations, every 10000 steps. Score averages the last two evaluations, then the three seeds.
- Ties: settings within **0.005** of the minimum mean; lowest population seed standard deviation, then smallest grid step, then stable enumeration if still identical. This operational rule was written **2026-11-06 before tuning**.
- Selected **2026-11-06**: SARSA step 0.03/11 and λ 0.5; RegQ step 0.01 and regularization 0.001; myopic step 0.03 and potential scale 50 s. All use constant exploration 0.05. No later result changes them.
- Every official arm gets **2000000 steps × seeds 0–9** and 20 validation evaluations every 100000 steps. Deploy the mean of post-update finite weights over the final **10%** of training. Final weights are secondary and never selected.
- Divergence is a nonfinite weight or norm above 10000. Preserve/flag the seed. Before the averaging window use the last finite weights; inside it average finite window weights. A softmax fallback was conditional on more than half the RegQ tuning runs diverging; it did not trigger.
- A 16-feature exploratory SARSA arm inherits the primary setting without retuning, uses validation only and never enters hypotheses or test.

Step normalization is asymmetric: SARSA divides by feature count, RegQ/myopic do not. This remains a limitation; there was no result-driven corrective search. Potential ablations inherit linear-arm hyperparameters rather than receiving independent tuning.

## Dated amendments and predictions

**2026-11-05, before task results:** align feature selection with validation; remove the reduced-seed exception; revise the RegQ grid to {0.001,0.01}; adopt final-10% averaging for every learner, finite-weight handling and secondary final-weight reporting. Evidence was synthetic correctness tests and consistency with the frozen protocol. Budgets and confirmatory thresholds were unchanged.

**Before tuning**, recorded **2026-11-06**, interpreting the **2026-11-05** feature control: H1′ was expected not to meet its threshold; SARSA–myopic differences might be small in their shared linear policy class.

**2026-11-06, after tuning and before official training:** the tuning scores suggested a larger difference, questioning the prior small-gap expectation. This is a dated interpretation, not a revised hypothesis.

**2026-11-06, before test opening:** official validation suggested H1′ and H-forward would not meet their thresholds. H-shaping final differences were near the closeness boundary, but its validation area range was below 10%, so the combined criterion was also expected not to hold. The original small-gap prediction agreed with official validation; the post-tuning reversal did not. RegQ's stronger validation mean was descriptive and did not change the primary method.

## Single test opening and closure

After official training/policy freezing, the authorized campaign included all six canonical arms, tail-average and secondary final policies, and canonical references on 3 test parts × 5 variants. It excluded the full-feature exploratory arm. Test involved no training or selection.

The first attempt stopped while hashing a missing nominal point-cache input, **before environment construction or performance information**. Deterministic nominal-only input preparation and manifest checks preceded one separately authorized continuation, completed **2026-11-07**. The initial claim was retained; an exclusive continuation claim prevented a second continuation. The initial failure did not enable adaptive performance-based reuse.

The holdout is now **closed**. Editorial work, numerical exports, software tests and viewer demonstrations do not authorize reopening it. Later research requires a separately designed evaluation policy.

## Statistics

Base part is the statistical unit. Primary presentation shows each test part, averaging its variants/seeds. Secondary presentation averages part means. Existing intervals used **10000** repetitions: resample parts, variants within part, seeds within variant; per-part intervals omit part resampling. Call these **hierarchical bootstrap intervals**, not guaranteed confidence coverage. Matching-variant differences are descriptive.

Three test parts are insufficient for significance claims. Variants/seeds are not independent extra parts. No significance tests or multiple-testing corrections are used. Public report preparation reuses stored interval endpoints and performs no bootstrap resampling.

## Model limitations (reported with every result)

- Ideal sensor: binary measurement; incidence-related precision loss, shiny-surface multiple reflections, lower laser-line-end accuracy and measurement noise are not modeled. Camera field of view covers the whole line. The analytic sweep neglects discrete profile spacing.
- Single setup without a turntable: interior and table-facing surfaces cannot generally be measured externally. Coverage is relative to reachable union, not all surface area.
- Reachability/static capsule and sensor-box clearance are checked at five sweep samples. Transfers, between-sample motion, self-collision and fixtures are not certified collision-free.
- Cycle time excludes data processing, registration and CAD comparison.

The later recorded-path audit used train/validation only with unchanged geometry. Under the same unwrapped endpoints, rest-to-rest assumptions and speed/acceleration limits, direct move time lower-bounds a collision-free detour. Extra avoidance time was not measured. Test collision statements are analogies from train/validation, not fresh test measurements. Learner test step traces were not stored; independent validation checked records/metrics rather than replaying policies.

## Integrity

Recorded playback is portable; live validation requires original caches/records. Published linear weights contain only tail averages. Original results, manifests and demo bytes are unchanged. Historical source identifiers in those immutable records remain provenance; editorial history changes do not recalculate results.
