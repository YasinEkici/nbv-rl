# Plan 002: Phase 1 — classical RL agents

Status: completed
Opened: 2026-11-03   Closed: —
Related decisions: D-038, D-041, D-043, D-045…D-063

Sections 1–3 were written on 2026-11-03, before any agent code and before any agent result, and approved by the user on 2026-11-04, including every item marked **Proposal**. Must finish before the midterm presentation (PROJECT.md: weeks 4–7). PROTOCOL v1 (frozen, D-043) is not changed by this plan.

## 1. Goal
Does a linear-function-approximation RL agent (no neural networks) learn a scan-pass policy that generalizes to unseen part types and closes a meaningful part of the cycle-time gap between the best myopic heuristic (greedy-IG/cost) and the industry-type CAD planner (ILP + TSP)?

## 2. Predeclaration (fill BEFORE any result is seen)

### 2.1 Methods (NumPy only, code in src/nbv/agents/)
- **Main: true online SARSA(λ)** with linear function approximation (van Seijen et al., JMLR 17(145), 2016), dutch traces, ε-greedy over unmasked actions. Why: with linear features it learns as fast or faster than conventional SARSA(λ), needs no choice of trace type, and is more robust to the step size.
- **Comparison 1: linear Q-learning** with the same features, L2-regularized (RegQ, Lim & Lee), same ε-greedy behaviour policy. The weight norm is logged at every evaluation; divergence (NaN/inf or ‖w‖₂ > 10⁴) is reported explicitly.
  - **Proposal: RegQ rather than the norm-scaled softmax** (Meyn 2024; Liu et al. 2025). Reason: RegQ keeps the behaviour policy identical to SARSA's (ε-greedy), so the comparison isolates the update rule (on-policy vs off-policy); the softmax variant changes exploration and the update together. The softmax variant stays a fallback, used only if RegQ diverges in more than half of the tuning runs (decided at the end of the tuning slice, before any official run).
  - A diverged official seed is kept and flagged. If divergence precedes the final 10% window, use the last finite weights; if it occurs inside the window, use the mean of the finite weights collected in that window (D-046).
- **Comparison 2: myopic agent** (contextual-bandit equivalent), same features, no bootstrapping.
  - **Proposal (correction to "γ = 0")**: with the objective-only reward (D-038), γ = 0 is degenerate: the immediate reward is −(t_move + t_scan) and contains no coverage term, so a γ = 0 learner would pick the fastest pass. The myopic agent therefore regresses on the one-step shaped reward with linear Φ computed at discount 1: target = r + Φ(s′) − Φ(s) (+ B at success), no bootstrap. It is a learned greedy-IG/cost and answers "does looking further ahead help?".
- **Ablation: Φ for true online SARSA(λ)** — none / linear / convex / value_estimate (D-038), all with the hyperparameters tuned for the main arm (caveat: this favours the Φ used in tuning).
  - **Proposal: the main arm (and Q-learning) uses Φ = linear** (α = 50 s), chosen now, before any result; it is the v2 shaping whose scale was calibrated in D-030.
  - **Proposal for value_estimate's speed**: cache Φ in the wrapper by (variant id, used-pass bitmask, last pass). Φ depends only on the measured set (determined by the variant and the used passes), the used mask and q mod 2π (determined by the last pass), so the key is exact; the cache stays agent-side (keys from `info["agent"]` and the observation). Seeds run in parallel. This arm uses all 10 official seeds and the same budget, or is dropped entirely by the predeclared time-shortage rule; there is no five-seed exception (D-046).
- **Discount**: γ = 1 for SARSA and Q-learning (episodes ≤ 24 steps; the return in PROTOCOL v1 is undiscounted; PBRS invariance with discount 1 needs no matching of discounts). **Proposal**: drop γ from the tuning grid for this reason.
- **Reward scaling (agent-side)**: rewards and Φ multiplied by 0.01 (returns become O(1)); the optimal policy is unchanged.

### 2.2 Features φ(s, a) (src/nbv/features.py; observation + nominal model only)
Computed for every unmasked pass a. All features are dimensionless and part-independent: fractions of the nominal reachable union, times divided by a fixed 10 s scale, or ratios to the best available pass in the same state. No per-part statistics (no fitted normalization), so a new part needs nothing but its CAD.

| # | Feature | Definition | Why |
|---|---|---|---|
| 1 | bias | 1 | intercept |
| 2 | gain | nominal new samples of a / \|nominal union\| | coverage contribution |
| 3 | gain_rel | gain / max gain over unmasked passes | scale-free ranking across parts |
| 4 | t_move | movej estimate from the current joint state ([−π, π]) / 10 s | motion cost |
| 5 | t_scan | scan time / 10 s | scan cost |
| 6 | rate_rel | (gain / (t_move + t_scan)) / max over unmasked passes | the greedy-IG/cost score, normalized |
| 7 | finishes | 1 if gain ≥ nominal coverage still needed to 95% | ends the episode early (bonus) |
| 8 | t_home | movej estimate from the end of a to q_home / 10 s | relevant for the last pass |
| 9 | next_rate | best gain/time rate after a, from the end of a over remaining unmasked passes, divided by the CURRENT state's best rate (feature 6's denominator); 0 if none remain or the current best rate is 0 | quality of the next option relative to the current best option |
| 10 | remaining | 1 − measured share of the nominal union | state: how much is left |
| 11 | needed | max(0, 0.95 − measured share) / 0.95 | state: distance to the target |
| 12 | step_frac | step / K_max | state: budget used |
| 13 | gain × needed | interaction | gain matters less near the target |
| 14 | finishes × t_home | interaction | cost of ending at a far pass |
| 15 | t_move × remaining | interaction | motion matters more when much is left |
| 16 | elevation | sin(elevation of a's view direction) | cell geometry (low passes are slower to reach) |

The table defines the full 16-feature reference. The D-046 check selected **11 features** on VAL (D-049): retain 1–8 and 10–12 for all subsequent tuning and retained learner arms; remove 9 and 13–16. The full implementation stays available for the recorded comparison and correctness tests. Speed: gains use the class compression of `nbv.shaping` (unique nominal visibility columns), so feature 9 costs O(P² · classes) per step when requested.

### 2.3 Hypotheses (PROTOCOL v1 metrics; test parts only in the final evaluation)
Reference values (val, canonical, results/baselines_v1.md): normalized cycle time greedy-IG/cost 1.186, MCTS 1.104, ILP + TSP 1.000, joint optimum 0.984.

- **H1′** True online SARSA(λ) closes at least half of the normalized gap between greedy-IG/cost and ILP + TSP.
  - Val reference threshold: ≤ (1.186 + 1.000) / 2 = **1.093** (better than MCTS on val, 1.104).
  - Test criterion (relative, because the threshold must come from the same parts): for test part p, threshold_p = (1 + g_p) / 2, with g_p = greedy-IG/cost normalized cycle time on part p from the predeclared test baseline run (2.4). Supported if the mean over the 3 test parts is ≤ the mean threshold and at least 2 of 3 parts meet their own threshold.
  - If rejected: report per test part the fraction of the gap closed, (g_p − x_p) / (g_p − 1), and the position relative to MCTS. Expectation from the literature: RL agents are comparable to benchmark heuristics but weaker than near-optimal analytic solutions (Kaiser 2025); learned heuristics approach solvers without usually beating them (Kool et al. 2019).
- **H-forward** True online SARSA(λ) has a lower normalized cycle time than the myopic agent.
  - Supported if the mean over test parts is lower by ≥ 0.02 and the paired difference (matching variants, seeds averaged per method) is negative on at least 2 of 3 test parts.
  - If rejected: report the paired difference per part; conclude that one-step lookahead features already capture what bootstrapping adds.
- **H-shaping** The choice of Φ changes learning speed but not the final performance markedly.
  - Interpretation: none versus linear compares initial values when Φ is in the linear feature space; it is not a distinct learning objective (Wiewiora 2003; equivalence tested in Slice 2).
  - Learning speed: area under the val learning curve (normalized cycle time over environment steps; lower is better). "Changes" = best and worst Φ differ by ≥ 10% in this area.
  - Final performance: "not markedly" = every Φ arm within 0.02 of the linear arm in mean normalized cycle time on the test parts (0.02 ≈ 1 s of a ~50 s cycle, the size of the joint optimum's advantage over ILP + TSP and of typical per-part interval widths).
  - If rejected: report which Φ helps or hurts and by how much; note that PBRS guarantees policy invariance, not learning speed (Müller & Kudenko 2025; Canonaco et al. 2024).
- **H3** (camera-only model overestimates coverage): measured in Plan 001 (results/incidence_sensitivity_v2.csv); reported descriptively, no new run.
- Q-learning has no hypothesis: reported descriptively with divergence statistics.

Evaluation: per test part first (mean over variants and seeds with its bootstrap interval), then the split mean with the hierarchical bootstrap interval; paired differences with `nbv.evaluation.paired_difference` (this resolves the Open issue; `iqm` is deleted at closure if still unused). No significance tests (PROTOCOL v1).

### 2.4 Tuning protocol (against maximization bias; Patterson et al., JMLR 25(318), 2024; Eimer et al., ICML 2023)
- Tuning seeds 1000–1002, disjoint from the official seeds 0–9 (PROTOCOL v1); a test checks the disjointness.
- Equal tuning budget: every learner gets 12 grid points × 3 tuning seeds × 2·10⁵ steps (10% of the official budget).
- **Proposal: grids (12 points each)**:
  - SARSA(λ): step size α ∈ {0.01, 0.03, 0.1} (divided by the number of features) × λ ∈ {0.5, 0.9} × ε schedule ∈ {linear 0.2 → 0.01 over the first 50% of the budget, constant 0.05}.
  - Q-learning: α ∈ {0.01, 0.03, 0.1} × L2 weight η ∈ {10⁻³, 10⁻²} × the same two ε schedules (D-046).
  - Myopic: α ∈ {0.01, 0.03, 0.1} × shaping scale α_Φ ∈ {25, 50} s × the same two ε schedules (α_Φ sets its coverage-versus-time trade-off).
- One setting for all parts (no per-part tuning). Training on train variants, sampled uniformly per episode from an environment seed sequence that is the same for all methods (paired); agent seed and environment seed are separate.
- Selection rule (written now): lowest mean over tuning seeds of the val normalized cycle time, averaged over the last 10% of the evaluations (ε = 0, all 15 val variants); ties (within 0.005) → the lower spread across tuning seeds, then the smaller step size.
- Selected in Slice 4 (D-053): SARSA grid alpha=0.03 (effective 0.03/11), lambda=0.5; RegQ alpha=0.01, eta=0.001; myopic alpha=0.03, Phi scale=50 s. All three use constant epsilon=0.05. The 11-feature selection is unchanged. Machine-readable settings and full evidence: results/selected_settings_v1.json and results/tuning_v1.md, generated from registry rows.
- Official runs: the selected setting, seeds 0–9, 2·10⁶ steps per seed (PROTOCOL v1), training on train; every 10⁵ steps an ε = 0 evaluation on all val variants (learning curve, 20 points). The official policy is the mean of post-update finite weights over the final 10% of training steps, for every learner (D-046); no checkpoint selection. Inside this window, evaluations use the running tail mean. Final-weight evaluation is secondary appendix material and never selects anything. Divergence handling is in 2.1.
- Test: after the official runs are complete and frozen, one evaluation of every arm's tail-mean policy and secondary final-weight policy (seeds 0–9) and one run of the canonical baseline set (D-041) on the 3 test parts × 5 variants, from the same commit. Nothing touches the test parts before.

### 2.5 Runs and outputs
- Exploratory extra arm (D-050): full 16-feature true online SARSA(λ), no separate tuning, the selected main-arm setting, Φ=linear, seeds 0–9, 2000000 steps on train with VAL evaluation only. Never evaluated on test, never used in a hypothesis. It diagnoses whether the short-budget D-049 result reflects representation or insufficient learning. Drop this arm first if time is short, before value_estimate.
- Runs: tuning grid (exploratory registry rows, tag p002t); official SARSA(λ), Q-learning, myopic with Φ = linear, and SARSA(λ) with Φ ∈ {none, convex, value_estimate} (canonical rows, tag p002, `budget_steps` = 2000000); test evaluation (canonical rows, split test) and the test baseline run.
- Seeds / budget: tuning 1000–1002 at 2·10⁵ steps; official 0–9 at 2·10⁶ steps for every retained arm.
- Outputs: registry rows; val learning curves (seed mean and range) as figures under results/; RL rows added to the main table in the format of results/baselines_v1.md (results/agents_v1.md); Φ ablation table; Q-learning weight-norm and divergence report; wall time and steps/s per arm.
- Estimated compute (to be measured in Slice 3): ~1 000–2 000 agent steps/s per core → tuning ~6 CPU-h, official runs ~35 CPU-h; ~2 h wall time on 20 cores.
- Comparisons to report: H1′, H-forward, H-shaping as above; Q-learning descriptively.
- This plan will NOT: use neural networks or RL libraries; change the environment, reward, metrics or split (PROTOCOL v1); tune per part; select anything on test parts; run 128 candidates; start Phase 2.

### 2.6 Amendment log (before any task result)
2026-11-05, user-approved, D-046; evidence is limited to Slice 1 unit tests and compliance with frozen PROTOCOL v1:
- Feature selection: replace "train only" by training on train and evaluation/selection on VAL (PROTOCOL v1 selection rules).
- Remove the five-seed value_estimate exception: use seeds 0–9 at the full budget, or drop this arm entirely under the existing time-shortage rule.
- Clarify next_rate's denominator as the current best rate, not a newly normalized next-state maximum.
- RegQ η changes from {10⁻⁴, 10⁻³} to {10⁻³, 10⁻²}; still 12 grid points and equal tuning budgets. Both old values failed to stabilize the synthetic divergent example. Small regularization is deliberate: larger η can bias the solution (regq; Manek & Kolter 2022, secondary). The softmax fallback is unchanged.
- Replace final weights by the final-10% weight mean for all learners (Polyak–Ruppert tail averaging). Constant-step TD tail averaging reduces variance at order 1/k under the assumptions of samsonov2024; Slice 1's MRP averages improve over last weights. It does not remove constant-step Markovian asymptotic bias (huo2026; zhang_xie2024). These results concern policy evaluation/fixed behaviour, whereas SARSA changes policy: the low-ε tail is a heuristic, not a guarantee. Final weights are reported only as a secondary appendix comparison. No additional validation selection.
- Feature check predeclaration: 16 versus 11 features (remove 9, 13–16), true online SARSA(λ), α = 0.03/d, λ = 0.9, γ = 1, linear ε 0.2 → 0.01 during the first half, Φ = linear (50 s), reward scale 0.01; seeds 1000–1002, 200000 steps each, same episode seed stream, train variants only for learning. Evaluate all 15 VAL variants with ε = 0 every 10000 steps; final 10% evaluations are at 190000 and 200000 steps, using the running tail mean of post-update weights from steps 180001 onward. Compare the mean over those evaluations, then seeds, of normalized cycle time. Retain the five extras only if 11-feature minus 16-feature time is at least 0.01; otherwise drop them. No test parts. L1 selection (kolter2009) addresses large feature sets relative to data; a predeclared removal comparison suffices here with 16 features and millions of steps. Runs are exploratory and recorded with budget_steps before Slice 4.

### Gözlem (2026-11-05, ayar taramasından önce)
Recorded on 2026-11-06 before Slice 3 or the tuning sweep, interpreting the 2026-11-05 feature check; this observation does not revise any hypothesis or selection rule.
- In linear Q, the state-only features bias, remaining, needed and step_frac add the same value to every action and cannot change the ranking (also why the Wiewiora test works). The 11-feature policy is an argmax of a fixed-weight pass score using gain, gain_rel, t_move, t_scan, rate_rel, finishes and t_home: a learned generalization of greedy-IG/cost. Only finishes and the state-relative normalizations gain_rel/rate_rel can express an episode-dependent trade-off between those terms. State × action interactions 13–15 and lookahead 9 were removed by D-049.
- Prediction written before tuning: H1′ is likely to be rejected; the SARSA/myopic difference in H-forward may be small because they learn within the same policy class. The 11-feature check's VAL score (rounded 1.195) is already close to greedy-IG/cost (1.186); evidence: results/feature_check_v1.md, generated from the registry. This is not a hypothesis verdict.
- Interpretation: Phase 2 representation learning can learn state–action interactions.
- The exploratory 16-feature arm added in 2.5 is an explicit predeclaration addition (D-050), separate from this interpretation; it does not reopen D-049 or enter any hypothesis.
- Post-tuning observation recorded 2026-11-06, before official runs (D-054): SARSA grid alpha was divided by 11 (effective 0.0009–0.009), whereas RegQ/myopic used 0.01–0.1 directly. They never tried SARSA's selected effective step 0.03/11. This asymmetry may favour SARSA in H-forward (Patterson et al. 2024, designer's curse) and must be a limitation in the report/presentation. Selected-tuning train prefixes provide empirical E[‖x‖²] and alpha × E[‖x‖²] in results/regq_diagnostic_v1.md, generated from registry rows. Inverse feature-norm step scaling is a common TD practice (active-feature count for binary features). No new tuning or setting change: myopic's small-alpha tuning scores are similar and RegQ is insensitive in discrete score, so there is no present sign that a smaller step would reverse the comparison; this is an interpretation, not evidence of no effect.
- Updated prediction, 2026-11-06, after tuning and before official runs: the earlier expectation of a small H-forward gap does not seem to fit the tuning table (results/tuning_v1.md). A possible explanation is myopic's additive fixed exchange rate, Phi scale × gain minus time, versus greedy-IG/cost's ratio. Weighted set-cover greedy chooses lowest cost per newly covered element (Chvátal 1979, secondary via Young); budgeted maximum coverage also uses the uncovered-weight/cost rule (Khuller et al. 1999). The latter paper's 1−1/e guarantee needs partial enumeration, not the ratio rule alone; neither guarantee transfers directly to state-dependent robot travel costs. No analogous guarantee is claimed for the additive rule. This is not a hypothesis verdict: official runs and the later single test evaluation decide.
- Curve diagnosis recorded before test opening (2026-11-06): the official SARSA/linear mean at the tuning budget is already better than its selected tuning score, then becomes worse and oscillatory later. Under the user-specified diagnostic comparison, bounded-policy oscillation is the dominant explanation for the final deterioration; tuning selection optimism can coexist but does not explain an initially better official score by itself. See the registry-generated checkpoint/seed-range and AUC table in results/pretest_v1.md. The tuning score uses tail averaging while the early official point precedes the official tail window, so this comparison does not identify a unique cause. Smith & Winkler 2006/eimer2023 motivate selection optimism; zhang2023_sarsa and its secondary Gordon references motivate bounded chatter. Their assumptions are not claimed as a guarantee for our epsilon-greedy, constant-step true-online implementation. Last versus tail-policy differences and RegQ's steadier curves support this interpretation. No new run or selection.

#### Cross-split observation (2026-11-07; after the closed test, not a decision)
The value_estimate arm has the lowest SARSA mean on all three splits and is close to MCTS on test. This was not a predeclared hypothesis. A best-of-six test-only claim risks the optimizer's curse; the supporting observation is consistency across splits, not a new selection. The mechanism is unverified: a value heuristic might reduce state-value variation and ease learning action distinctions (cheng2021_hurl, wang2016_dueling, canonaco2024). Plan 003 may test this explanation on a separately declared design. Main SARSA/linear stays fixed; H-forward is rejected at full precision (D-065).
<!-- BEGIN GENERATED: cross_split -->
| SARSA Phi | Train | VAL | Test |
|---|---:|---:|---:|
| linear | 1.196642 | 1.207170 | 1.252614 |
| none | 1.196512 | 1.206435 | 1.244123 |
| convex | 1.198637 | 1.198957 | 1.227824 |
| value_estimate | 1.162074 | 1.187515 | 1.142396 |
| MCTS reference | 1.094257 | 1.104443 | 1.144764 |

H-forward paired SARSA-minus-myopic difference: -0.019894820622543; missing improvement to the predeclared 0.02 threshold: 0.000105179377457. No rounding-based reinterpretation.
<!-- END GENERATED: cross_split -->

### Test öncesi tahmin (2026-11-06)
Recorded before test opening, using official VAL only; interpretation, not a decision or a selection amendment.
- H1′: expect rejection. SARSA VAL=1.207170, threshold=1.093175, greedy-IG/cost=1.186349; the main method remains true online SARSA(lambda).
- H-forward: expect rejection. SARSA−myopic=-0.012366, short of the required −0.02. The original pre-tuning small-gap prediction matches official VAL; the post-tuning reversal does not. It illustrates the optimizer's curse/overinterpretation of a selected tuning estimate, while A2's time-series evidence supports later oscillation as the dominant final-score deterioration mechanism.
- H-shaping: final VAL differences remain near the 0.02 boundary; no arm or potential is selected from this observation.
  - none: final VAL minus linear=-0.000735; area=2389857.477368.
  - convex: final VAL minus linear=-0.008213; area=2325504.372596.
  - value_estimate: final VAL minus linear=-0.019655; area=2231180.231636.
  - linear area=2345030.439799; relative worst−best area difference=7.111808%, below the 10% speed criterion. Thus expect overall H-shaping rejection, despite borderline final closeness.
- RegQ: best VAL learner at 1.171129, the only final learner beating greedy-IG/cost. No hypothesis, no promotion to primary method.
- Full numerical checkpoint/range and area evidence: results/pretest_v1.md; machine-readable predictions: results/pretest_predictions_v1.json. These artifacts are registry-derived; final verdicts await the single test opening.

## 3. Slices
### Slice 0 — Planning freeze ✅ completed 2026-11-04
- Scope: this predeclaration approved; LITERATURE entries; project test-before-commit rule.
- Acceptance: user approval of sections 1–3; Status: active.

### Slice 1 — Agent core and correctness tests (tests first) ✅ completed 2026-11-04
- Files: src/nbv/agents/{linear.py (weights, ε-greedy over the mask), sarsa.py (true online SARSA(λ)), qlearning.py (RegQ), myopic.py}, tests/test_agents.py.
- Tests (written before the agents):
  - true online SARSA(λ) on a small MRP with a closed-form value: weights converge to the known solution; one hand-computed update matches; with λ = 0 the updates equal one-step SARSA;
  - a masked action is never chosen (also under exploration);
  - same agent seed and environment seed → identical run; different agent seed → different run;
  - no NaN, and the weight norm stays bounded on the synthetic environment; Q-learning's divergence check fires on a constructed divergent case;
  - tuning seeds and official seeds do not overlap.
- Acceptance: tests pass; tests/test_privileged.py still passes with the new modules.
- Result: 36 tests in tests/test_agents.py, written before the agents. The weights of true online SARSA(λ) equal a brute-force online λ-return algorithm for λ ∈ {0, 0.5, 0.9, 1}, γ ∈ {0.9, 1}. Agents receive only feature matrices, masks and rewards; the step size is passed as given (the division by the number of features of 2.4 belongs to the harness, Slice 3).
- Review accepted 2026-11-05: the λ = 1 convergence measurement uses the last-half weight mean instead of last weights; the RMSE threshold tightened from 0.05 to 0.02 because constant-step noise persists in last weights.

### Slice 2 — Features ✅ completed 2026-11-05
- Files: src/nbv/features.py (state-action features), tests/test_features.py.
- Tests: shape (P × 16), finite and bounded values, invariance to the real part (privileged test), part independence (scaling a part's time matrices by 2 leaves the ratio features unchanged), speed (target ≥ 3 000 feature evaluations/s).
- Feature check: learn on train and select on VAL by the frozen rule in 2.6 (D-046); record the outcome before tuning. The smallest reusable training loop belongs to agents/ and will be extended in Slice 3.
- Implemented: 16/11 nominal-only features, sample-level reference and time-scaling tests, C = 1 − remaining with no clipping, and Wiewiora equivalence for λ in {0.5, 0.9, 1} over at least 3000 steps each (atol 1e-9). Thread tests first failed 2/2, feature/averaging/equivalence tests first failed 11/11, loop tests first failed 3/3. Loop code is in agents/train.py; scripts/feature_check.py is a thin entry point.
- Throughput acceptance: `uv run python scripts/feature_check.py --benchmark` measures complete 48 × 16 matrices at reset on train parts 46262, 65149, 124336; 100 warm-up calls and 10000 measured calls per part, initialization excluded. Float32 class counts are exact for the 20000 samples and speed up overlap calculations. Measurements and the check outcome follow below.
- Final throughput: 5513.3 / 3369.8 / 3442.3 complete matrices/s on those three train parts, respectively (all ≥3000). Reusing the already computed union-class counts avoids a second sample reduction; C remains exactly the complement of remaining.
- Feature check completed from clean source commit 56429e6: six exploratory registry rows `p002fc-sarsa{16,11}-val-s{1000,1001,1002}-v4-*`, each budget_steps=200000. Every paired episode-seed prefix matched, each tail contained 20000 finite post-update weights, all seeds remained finite, and no test variants were evaluated. Table generated by `uv run python scripts/report.py --feature-check --out results/feature_check_v1.md`; numerical results are in the registry and generated table, not copied here. D-049 selects 11 features by the predeclared threshold.
- At Slice 2 closure the reusable loop supported only the feature check; Slice 3 and tuning were then pending approval. They were subsequently authorized and completed below. Slice 2's final implementation test run before the check was 189 passed.

### Slice 3 — Training harness ✅ completed 2026-11-06
- Files: extend src/nbv/agents/train.py (episode loop, periodic val evaluation, tail-mean policy and secondary final weights in outputs/, registry rows with `budget_steps`), scripts/train_agent.py (thin), value_estimate cache in src/nbv/shaping.py, tests.
- Acceptance: a 10⁴-step smoke run per method; measured steps/s per arm; value_estimate cache speed-up measured.
- D-051: reuse the existing loop; 20 evaluations for either budget (the final two select settings), learning/evaluation timing, explicit train/VAL-only cache-content hashes, tail and final weights, curves and divergence. scripts/train_agent.py orchestrates parallel seeds; --smoke runs every method × Phi and an uncached value_estimate control per method on one core, plus full-feature linear SARSA for the 70-seed time estimate. No official runs are authorized in this turn.
- Stop gates: any failed test or nonfinite tuning update/score; >12 h official estimate at 20 cores; >18/36 RegQ tuning runs diverged (stop without implementing softmax).
- Selection implementation (before tuning): spread = population standard deviation across the three tuning seed scores; tie band = all settings within 0.005 of the minimum mean, then lowest spread, then lowest grid alpha (D-051).
- Acceptance met (D-052): 16 exploratory smoke rows p002smoke-*, each budget_steps=10000: 12 method × Phi combinations, 3 uncached value_estimate controls and full-feature linear SARSA. No divergence or nonfinite learner values. Single-core throughput, cache speedups and the 70-seed/20-core official-time estimate are generated from registry rows in results/training_speed_v1.md (`uv run python scripts/report.py --training-speed`). The estimate passes the 12-hour gate. No official runs were started; Slice 4 is authorized next.

### Slice 4 — Tuning grid ✅ completed 2026-11-06
- 3 methods × 12 points × 3 seeds × 2·10⁵ steps (exploratory). Selection by the rule in 2.4, recorded in a DECISIONS entry with the table.
- `uv run python scripts/train_agent.py --grid --workers 20`; tag p002t, features=11, Phi=linear (myopic scale is its grid setting). Tables and selected settings are generated only from registry rows by `uv run python scripts/report.py --tuning`; success rates accompany normalized times. No official or test runs.
- Completed all 108 exploratory runs from clean source commit acad3a2: 3 methods × 12 settings × seeds 1000–1002 × 200000 steps. Every run has 20 VAL evaluations and 20000 finite tail weights; all 165 eligible cache inputs share one unchanged content hash. The 36 runs for each seed share identical environment-seed prefixes. No numerical divergence or nonfinite learner update/score; RegQ fallback did not trigger. Some finite policies have censored episodes, shown by their success rates in the table.
- D-053 records the selected settings. SARSA's minimum-mean setting is selected directly; RegQ and myopic use the lowest-spread setting inside the predeclared 0.005 band rather than the absolute minimum mean. Grid tables, seed scores, spread, success and divergence: results/tuning_v1.md; exact selected parameters: results/selected_settings_v1.json. No post-result grid change, extra tuning or test evaluation.
- Stop here for user approval before Slice 5, including the exploratory full-feature arm. These are tuning scores, not official performance estimates or hypothesis verdicts.

### Slice 5 — Official runs, Φ ablation, test evaluation — ~1.5 days
- Official runs (canonical), learning curves, then the single test evaluation of agents and baselines; tables via scripts/report.py; hypothesis verdicts.
- Slice 5a authorized 2026-11-06: first diagnose saved RegQ tuning weights and alpha scaling (D-054), then run 60 canonical train/VAL seeds plus 10 exploratory full-feature seeds, all at 2000000 steps with D-053 settings. Never open test parts in this slice. Stop on unexpected post-implementation test failure, nonfinite learner values, >5 diverged seeds in any arm, or total official elapsed time >6 h. Deliberate pre-implementation red tests are required and do not trigger STOP.
- Prepare the separate one-time test CLI with explicit opening flag, clean tree, complete official-set/checkpoint checks, no previous test row, and an exclusive persistent opening claim. Test guards with synthetic registries only (D-055). The CLI and actual test caches remain unopened until separate user approval.
- **Slice 5a completed 2026-11-06 (D-056):** all 60 canonical and 10 exploratory runs completed from clean source efe68c6, each at 2000000 steps with 20 VAL evaluations and 200000 finite tail weights. No divergence; shared explicit cache-content provenance and paired episode-seed prefixes verified across all arms. Both policy-weight kinds, learning curves and episode metrics remain in outputs/; registry tags p002/p002full identify all runs and p002timing records total wall time, below the six-hour gate.
- Registry-driven VAL report: results/official_val_v1.md, per-part tail/final table results/val_summary_v1.csv, complete mean/min–max curves results/val_curves_v1.json, individual arm and combined official PNG/PDF figures in results/figures/. Regenerate with scripts/report.py --official-val. Final-weight success is shown separately from tail-policy success; no reselection on these results.
- **Slice 5b awaits user approval:** scripts/evaluate_test.py is prepared but was not run. Synthetic guards include the real output writer's YAML format, actual final-step history, finite/matching checkpoint files and persistent single-opening protection. All 60 canonical VAL checkpoints pass the same completion check. No test rows or opening claim; no test variant loaded. Hypothesis helpers tested on synthetic values only; no verdict yet.
- Slice 5b authorized 2026-11-06: before test opening fix identity/manifest guards (D-057), correct the inapplicable RegQ formula (D-058), invalidate historical diagnostic policy summaries without deleting rows (D-059), and defer live divergence stopping explicitly (D-060). Preserve original outputs, all selected settings/arms and SARSA as the primary method. Evaluate train without learning; commit the artifact manifest, curve diagnosis and pretest predictions before the one-time test opening. Any unexpected opening/evaluation error or incomplete campaign stops without retry. Slice 6 remains pending separate approval.
- Pretest work completed (D-061): manifest dd1e0fc; VAL predictions ed0b920; 120 canonical train evaluation rows (60 checkpoints × tail/final policies), all 40 train variants, 4800 episodes. Train table results/train_v1.md; per-episode metrics stored at full precision in registry extra_json. Original source checkpoints still match the committed manifest. No test row or opening claim at this checkpoint.

**Slice 5b completed 2026-11-07 (D-062/D-063):** the initial opening stopped before environment construction; the user authorized one guarded continuation after nominal input preparation. All 145 canonical evaluation rows completed from clean source 60f6f33, on all 15 test variants, with both source weight kinds and the D-041 baseline set. Immutable claims/manifests and exact campaign counts are audited in results/test_execution_v1.json. The earlier "awaits approval/no claim" bullet above is the historical Slice 5a checkpoint, superseded by this completion. Test tables, secondary metrics, hypothesis rejections and pretest-prediction comparisons are registry-generated in results/test_v1.md; no settings or primary-method change. Slice 6 remains pending user approval; Plan 002 stays active.

### Slice 6 — Closure and README — ~1 day
- Closure, README at report level (requested by the instructor), results tables and figures linked from the registry.
- Completed 2026-11-07 (D-064–D-067). Earlier "pending approval" statements above are historical slice checkpoints; this closure supersedes them without changing their recorded predictions or results.

If time runs short: drop the exploratory 16-feature arm first, then value_estimate, then Q-learning; the main arm, the myopic agent and the test evaluation stay.

## 4. Open issues
All unresolved items are transferred with reasons to [Plan 003 Open issues](../003-deep-rl-and-ablations.md): live divergence stopping, future learner step traces, RegQ guarantees, constant-step bias, information-boundary checks, PROTOCOL v2 tests, conditional casting-source/IK questions, and representation/guidance/residual options. Phase 1 infrastructure/feature-selection items were resolved in their slices; legacy records remain preserved. The unused iqm helper was removed at closure. No issue justifies another Phase 1 test opening.

## 5. Closure
- All three predeclared hypotheses are rejected under the unchanged rules; H-forward narrowly misses its improvement threshold. Exact effects and if-rejected reports are in results/test_v1.md. SARSA/linear remains the primary method.
- value_estimate has the lowest SARSA mean across train/VAL/test and is close to MCTS on test: a descriptive observation, not a selected replacement or a confirmed mechanism (D-065).
- Official curves support bounded oscillation as the main descriptive account of late SARSA deterioration; tuning optimism may coexist. The pre-tuning small-gap prediction fits the official result; its post-tuning reversal did not. No convergence theorem is claimed for this implementation.
- RegQ is empirically stable and competitive among main learners, with no evaluated theoretical stability condition for eta I (D-058). All original retained runs remain in the record.
- Tail averaging moderates last-weight instability and preserves the primary/secondary distinction; the full final-weight appendix remains visible. It does not remove constant-step bias or uniformly improve every arm.

Registry rows added by this plan (automatically counted; full run_ids in [plan002_registry_v1.csv](../../../results/plan002_registry_v1.csv)):

| Tag | Status | Split | Rows |
|---|---|---|---:|
| feature_check | exploratory | val | 6 |
| p002 | canonical | val | 60 |
| p002diag | exploratory | val | 3 |
| p002full | exploratory | val | 10 |
| p002planning | exploratory | val | 1 |
| p002smoke | exploratory | val | 16 |
| p002t | exploratory | val | 108 |
| p002test | canonical | test | 145 |
| p002timing | exploratory | val | 1 |
| p002train | canonical | train | 120 |

The official p002 rows are the frozen canonical train/VAL learning runs; p002train rows evaluate both stored weight kinds without learning; p002test contains both weight kinds and the D-041 baseline set. The exploratory full-feature arm is VAL only. p002diag policy summaries remain invalid as D-059 records; their original rows are preserved. p002planning measures VAL inference time, with n/a quality fields.

- Decisions added/used for closure: D-045 through D-067; D-064 records validation limits, D-065 the descriptive cross-split observation, D-066 VAL-only planning time, D-067 closure. Relevant prior environment/protocol decisions stay active; PROTOCOL v1 was not edited.
- Docs updated: README.md, project status notes, docs/DECISIONS.md, docs/LITERATURE.md, project workflow notes, this archive and draft Plan 003. Train/VAL/test reports carry the full model limitations; results/figures/ contains PNG/PDF presentation exports. Results tables and README blocks are generated from registered data, not transcribed.
- Validation: observe-red/green checks for report limitations, input reuse, VAL-only timing and citation inclusion; full pytest before every commit. Existing quality tables, settings, hypotheses, source artifacts and test claims were preserved. No test evaluation or real-test input preparation was run during Slice 6. Canonical quality data retain their original provenance.
- Handoff: Phase 1 complete; Plan 003 remains a draft awaiting the user's approval. No Phase 2 implementation, new holdout use, training or push. Local fast-forward and plan-branch deletion follow the project workflow.

### Slice 5b incident-1 input preparation (2026-11-07, before continuation)
- D-062 records the failed pre-environment hash stage at 4850144; the original opening claim remains unchanged. No test performance or registry test row was produced. The user explicitly authorized one guarded continuation, without any setting/arm/primary-method change.
- Stat-only inventory: results/test_input_inventory_before_v1.json and results/test_input_inventory_after_v1.json. Generated only missing nominal point caches and nominal ILP+TSP/joint plans; never constructed a test environment or evaluated a method. Exact generated paths, train point-content determinism evidence, plan sequences and solver status/bounds are in results/test_input_preparation_v1.json. D-033/D-035 settings remain fixed; all joint results prove the recorded integer-ms planner optimum, not an optimum of real-variant execution time.
- results/test_input_manifest_v1.json pins every required input with full SHA-256, including the original claim and incident record, source checkpoint files/configuration, nominal/variant visibility, nominal points, robot/planner inputs, d_plan and selected settings. The post-preparation inventory has no missing input. Test continuation remains gated until this manifest and recovery guards are committed and a fresh complete suite passes on the clean evaluation snapshot.
