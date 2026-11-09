# Evaluation Protocol

Status: FROZEN (D-043, 2026-11-03). Version: 1. Changes only through a new version with a DECISIONS entry.

Scope (D-040): version 1 covers Phase 1 (classical RL, Plan 002) and the baselines. The Phase 2 evaluation details (handling of SAC's continuous action and its coverage reference, which baselines run at 128 candidates, the Phase 2 budget) are defined as PROTOCOL v2 at the start of Plan 003, before any Phase 2 result is seen.

## Environment definition
- Information model (D-029): model-based; the nominal CAD is known. Passes, IK and planner time matrices come from the nominal mesh of each base part and are shared by its variants; the real variant determines what is measured and is never observed directly.
- Action: one of N candidate scan passes (N = 48; the 128-candidate set is the NSC reference, D-028, D-041). View directions: Fibonacci on the upper cap (elevation ≥ 10°) around the part's bbox center; standoff R(u) = h(u) + 150 mm (h = part support in direction u); sweep direction = global x projected onto the plane normal to the view; sweep length = part projection on the sweep axis + 2 × 20 mm (D-020). Sensor per D-019. Unreachable candidates (UR5 IK at 5 samples along the sweep, D-021; any robot body or the sensor head closer than 5 mm to the nominal part or the table at one of these samples, D-042) and already-used passes are masked.
- Measured surface sample (D-025): crossed by the laser plane within the sweep length; within the depth of field (150–250 mm) and the laser fan; laser incidence ≤ 65° (sensitivity at the visibility level: 60°, 70°; D-012) and camera viewing angle ≤ 65° (D-026); unobstructed to the laser source and to the camera (part and table as occluders). Surface = 20 000 area-weighted samples per base part, shared by its variants. Ablation: camera-only model (D-026).
- Coverage: measured samples / observable samples, where observable = union over the reachable candidates of that variant, for the candidate set in use (D-023).
- Robot cell (D-021, D-027, D-042; configs/robot_v2.yaml, configs/collision_v1.yaml): UR5 (Robotics Toolbox DH model), base at (−350, 0, 0) mm with yaw 180°, sensor origin 120 mm along the flange axis, joint limits ±360°, q_home = (0, −90, 90, −90, −90, 0)°. A pass is reachable if IK succeeds at 5 samples along its sweep with |Δq| ≤ 45° per joint between consecutive samples, every DH frame origin and the sensor origin stay ≥ 20 mm above the table, and every body (link capsules, sensor-head box) keeps ≥ 5 mm clearance from the nominal part and the table at every sample. Table exemption: the base, the shoulder housing and the first 65 mm of the upper arm sit at a fixed height above the table and are not table-checked. Switch to UR10 only if parts turn out too large, via a DECISIONS entry (D-011).
- Motion cost: URScript movej time in joint space from the end of one pass to the start of the next: trapezoid on the leading joint with v = 1.05 rad/s, a = 1.4 rad/s² (D-022). Scan time = sweep length / 100 mm/s. All reported cycle times come from the environment, which tracks the arm's joint state (per-joint [−π, π] transfer with ±360° limit check over the whole next pass); cached PTP matrices are planner approximations only, and every method's pass sequence (including the TSP tour) is re-evaluated in the environment before reporting (D-024).
- Episode (D-030, D-038): start at q_home; the episode ends at 95% primary coverage of the real reachable union, at K_max = 24 (48 candidates) / 40 (128, D-036), or when no pass is left; all three are terminal (terminated = True; truncated is never used, because the step count is in the observation). The return to q_home is part of the cycle time.
- Reward (D-038): the objective only: −(t_move + t_scan) per pass, −t_home at the end, + B = 250 s at the step that reaches 95%; the return is −cycle time + B·success. Coverage shaping is not part of the protocol: an agent may add potential-based shaping (`nbv.shaping.PotentialShaping`, Φ from the observation and the nominal model only, Φ = 0 at terminal states); its choice is an agent setting, reported with the agent.
- Relation to the assignment: the assignment's "positive reward for information gain" is the agent-side shaping term. Because Φ = 0 at terminal states, it cancels in the total return: it guides learning and does not change the objective.
- Information returned by the environment (D-038): `info["agent"]` may be read by agents; `info["eval"]` (real coverages) and the real variant (`env.data.real`) are for evaluation only (tests/test_privileged.py).
- ENV_VERSION: v4 (configs/env_v4.yaml, robot cell configs/robot_v2.yaml); bumped on any change above.
- Cycle time excludes data processing time (registration, CAD comparison) (D-036).

## Part split
- Base parts are split into train / validation / test by base part; all variants of one base part stay in one split.
- 14 base parts from Thingi10K (CC-BY / CC-BY-SA, D-015): 8 train / 3 validation / 3 test, stratified by hard-to-see ratio (D-016).
- Every base part is scaled to a 200 mm longest side (D-017).
- Variants per base part: 5; deviation generator seed: 0; deviations per D-031 (configs/parts_v2.yaml). Split manifest: configs/split_v1.yaml (generated from configs/parts_v1.yaml; the split is identical in v2).

## Metrics
- Coverage (D-028): primary = coverage of the own candidate set's reachable union (D-023); the 95% target and all cycle times use it. Secondary: NSC = coverage of the 128-candidate reachable union (common dense reference); raw = coverage of all samples. In v1 all methods run with 48 candidates; the 128-candidate set is only the NSC reference (D-041).
- Primary: estimated cycle time to 95% primary coverage (s), on test parts, presented per test part (see Statistics).
- Secondary (D-034): number of passes to 95%; primary coverage after 5 and 10 passes (final value if the episode ended earlier); AUC of primary coverage vs time on [0, 180 s], step function (coverage changes when a pass ends), divided by 180; total joint motion Σ|Δq| (rad) over all moves between passes, including from and to q_home; generalization gap = train − test in cycle time and in success rate. NSC and raw coverage after 5 and 10 passes are reported in extra_json.
- Open-loop methods also report the number of repair passes and the success rate without repair (D-032).
- Difficulty-normalized cycle time (D-034): normalized_cycle_time = cycle time / ILP + TSP cycle time on the same variant and candidate set. Raw times are always reported as well; the generalization gap and method comparisons are interpreted on this ratio, and the hierarchical bootstrap is applied to it.
- Censored episodes (D-014) are reported separately (success rate): an episode is censored if it ends without reaching 95%, either at K_max (or with no pass left), or earlier because the policy or the plan repair returns no action (None). Its time metric is the cycle time up to that moment plus the return home, with censored = true.
- cycle_time_to_95_s includes censored episodes (with that cycle time) and must always be reported together with success_rate. The per-episode censored flag is stored in episodes.jsonl; the registry has no separate censored column.

## Baselines
Definitions and solver settings in D-033; information model in D-029; canonical set in D-041. All canonical baseline runs use 48 candidates, train and validation parts (test parts only in the predeclared final evaluation).

Main table:
- Greedy-IG/cost (nominal visibility, move estimate + scan time); closed loop.
- MCTS: UCT on the nominal model at every real step; closed loop; seeds 0–9.
- Industry-type CAD-based planner: set-cover ILP (95% of the nominal reachable union, fewest passes) + TSP on ptp_pi, solved on the nominal CAD; open loop with repair. Set cover + TSP is a two-stage decomposition and is not exactly optimal for the joint objective (coverage and motion time). Reference of the normalized cycle time.
- Joint optimum (joint_opt, D-035): one CP-SAT model chooses and orders passes on the nominal CAD (coverage ≥ 95%, objective = scan + ptp_pi tour); a solver-based upper reference, not an oracle; open loop with repair; best solution and lower bound reported. Plans from scripts/joint_opt.py before scripts/run_baselines.py.

Appendix table:
- Random (closed loop, seeds 0–9).
- Fixed pattern: robot analogue of stepping a turntable in front of a fixed scanner; top pass + 8-direction ring at 45°, snapped to the nearest reachable candidate; open loop with repair.
- Greedy-IG (nominal visibility); closed loop.

Not canonical (D-041): greedy-oracle (real variant's visibility; a diagnostic only) and random + TSP (code kept, not run). No baseline runs at 128 candidates in v1.

Closed loop vs open loop (D-032): step-by-step methods run until 95% real coverage or K_max; plan-ahead methods are planned on the nominal CAD, executed in the environment, and repaired by cost-aware greedy on the nominal visibility if they end below 95%.

Planning difficulty: d_plan / d_plan95 = fewest passes covering 100% / 95% of the nominal reachable union (set-cover ILP), per base part, 48 and 128 candidates. It uses only the CAD and is computed for test parts too, but test values are never used in any tuning decision.

## Budget and seeds
- Phase 1 learners: 2·10⁶ environment steps per seed, seeds 0–9 (D-034). All Phase 1 learners get the same budget.
- Phase 2: defined in PROTOCOL v2 (D-040); the Plan 001 Slice 6 speed trial (scripts/phase2_speed_trial.py, Plan 003 → Open issues) is its input.
- Stochastic baselines: seeds 0–9; deterministic baselines: one run.

## Selection rules
- Hyperparameters, features, reward weights, checkpoints: validation parts only.
- Test parts: evaluated once per predeclared run.

## Statistics
- Unit: the base part (D-034, D-039). Primary presentation of test results: the value of each of the 3 test parts separately (mean over its variants and seeds, with a bootstrap interval over variants, then seeds). Secondary: the mean over parts with a hierarchical bootstrap interval (resample parts, then variants within a part, then seeds within a variant; `nbv.evaluation.hierarchical_bootstrap`), and paired differences on matching variants (`paired_difference`).
- Wording: "hierarchical bootstrap interval", never "95% confidence interval": with few parts it covers the true mean less often than its nominal level (simulation: ~72% at 3 parts, ~89% at 8; scripts/bootstrap_coverage.py).
- No significance tests and no Holm correction: 3 test parts are not enough for a significance test (with n = 3 the smallest two-sided paired Wilcoxon p-value is 0.25); results are descriptive. Variants and seeds are never counted as independent observations.
- Tables come from canonical registry rows through scripts/report.py (per-part table first, then the split summary).

## Sensitivity analyses
- Incidence angle (D-012): 60° and 70° at the visibility level only. Visibility of the train and validation variants is recomputed with the laser incidence limit at each angle (and, as a second variant, with the camera limit tied to it, D-026); reported: measurable surface share (reachable union / samples) and relative laser loss (H3). scripts/incidence_sensitivity.py → results/incidence_sensitivity_v2.csv. No method runs at 60° or 70°.
- Candidate density: the 128-candidate set is used only as the NSC reference (D-028); no method runs at 128 in v1 (D-041).
- Sensor model: camera-only ablation (D-026) gives H3, the relative laser loss.

## Model limitations (reported with every result)
- Ideal sensor (D-025): a sample is measured or not (binary); in reality precision falls gradually with the incidence angle (first order ~1/cos α). Multiple reflections on shiny surfaces, the lower accuracy at the ends of the laser line, and measurement noise are not modeled; the camera's field of view is assumed to cover the whole line. The analytic sweep itself only neglects the discrete profile spacing (0.1 mm at 100 mm/s and 1 kHz).
- Single setup without turntable (D-025, D-027): interior and table-facing surfaces are not measurable by any optical sensor from outside; coverage is relative to the reachable union (D-023).
- Reachability is checked at 5 samples along each sweep with a joint-step limit (D-021); transfers between passes are not checked. Robot–part and robot–table collisions are checked statically at those samples with capsules and a sensor-head box (D-042); transfers, self-collision and fixtures are not.
- Cycle time excludes data processing (registration, CAD comparison) (D-036).

## Correctness tests (must pass before any reported run)
- tests/test_visibility.py — analytic 60 mm cube (top pass covers the top face only); overhang where the camera sees but the laser is blocked (dual < camera-only); dual ⊆ camera-only per pass; determinism; samples lie on the variant surface.
- tests/test_env.py — coverage is monotonic; return = −cycle time + B·success; K_max ends the episode as terminal; a policy returning None ends the episode censored; info split into agent and eval fields; masking works.
- tests/test_robot.py — IK reproduces the sensor poses; PTP matrices; collision masks: masked passes have clearance < 5 mm and reachable ones ≥ 5 mm, the IK-only mask contains the reachable one, a collision config without the part mesh raises.
- tests/test_shaping.py — for every Φ, shaped return − environment return = −Φ(s_0); Φ = 0 at the end; Φ reads no privileged data.
- tests/test_privileged.py — features, baselines, shaping and agents never read the real variant or info["eval"] (static and run-time check); greedy-oracle is the only exception.
- tests/test_determinism.py — same seed → same episode.
- tests/test_split.py — no base part appears in two splits.
- tests/test_collision.py — collision diagnostic geometry (signed distance sign, capsule and box penetration, DH offsets).
- tests/test_baselines.py — exact k-pass max-coverage ILP on the real variant ≥ every method's coverage at k (k ≤ 14 on 2 train + 2 val variants); greedy-oracle ≥ greedy-IG at k = 1; Held-Karp = brute force; set cover minimal and feasible.
- tests/test_evaluation.py — AUC, coverage-at-k, hierarchical bootstrap; provenance: unique run_ids, outputs never overwritten, canonical runs refused on a dirty tree, line-ending-independent file hashes, cache manifests that follow content (D-037).

## Forbidden
- Using test parts for any selection.
- Comparing runs with different ENV_VERSION or split versions.
- Reporting single-seed results as final.
- Canonical runs from a dirty working tree (D-037).
- Significance claims from test-part comparisons (D-034).
