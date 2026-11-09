# Plan 001: Environment and baselines

Status: completed
Opened: 2026-10-28   Closed: 2026-11-03
Related decisions: D-002…D-044

## 1. Goal
Build a correct, fast, and tested NBV environment together with all baselines, and freeze docs/PROTOCOL.md.

## 2. Predeclaration (fill BEFORE any result is seen)
- Not filled before the runs. The baseline runs of this plan (Slice 5) are exploratory and were not predeclared; canonical runs follow after PROTOCOL v1 is frozen.
- The Slice 6 MaskablePPO run (scripts/phase2_speed_trial.py) was a speed measurement, not a learning experiment; no learning result was produced or reported.
- This plan will NOT: implement any RL agent (Phase 1 or Phase 2); evaluate on test parts.

## 3. Slices
### Slice 0 — Planning freeze ✅ completed 2026-10-28
- Approve or change D-002…D-008 in docs/DECISIONS.md. → All approved (`active`); D-008 notes that DQN vs. MaskablePPO is finalized in the Plan 003 predeclaration.
- Added D-009 (Python 3.14), D-010 (no rliable), D-011 (UR5), D-012 (incidence 65°), D-013 (part split), D-014 (censored episodes); values written to docs/PROTOCOL.md.
- Choose the part source and record licenses. → Deferred to Slice 1.
- Acceptance criteria: decisions marked `active` (or superseded); this plan set to `Status: active`. → Met.

Deferred protocol fields (still `<...>` in docs/PROTOCOL.md / PROJECT.md / README.md):
- Part source and licenses → Slice 1.
- `ENV_VERSION` → Slice 4.
- Phase 1 / Phase 2 step budgets → after the Slice 4 episode-time measurement, before the Slice 6 protocol freeze.

### Slice 1 — Parts ✅ completed 2026-10-29
- Scope: 14 base parts from Thingi10K (D-015), 8 train / 3 val / 3 test stratified by hard-to-see ratio (D-016, supersedes D-013), 5 variants each, deviation seed 0; 200 mm longest side (D-017); deviation generator (D-018); part split manifest; `tests/test_split.py`, `tests/test_parts.py` (determinism via SHA-256 of rounded vertices + faces). Fills PROJECT.md → Primary data and README → Data.
- Files to change: src/nbv/parts.py, scripts/prepare_parts.py, configs/parts_v1.yaml, configs/split_v1.yaml, tests/test_split.py, tests/test_parts.py, docs/PROJECT.md, README.md
- Commands: `uv run python scripts/prepare_parts.py fetch|split|variants --config configs/parts_v1.yaml`; `uv run pytest -q`
- Acceptance criteria: manifest is generated; tests pass. → Met: configs/split_v1.yaml generated; all 70 variants pass watertight / winding / volume checks.

### Slice 2 — Candidates and robot ✅ completed 2026-10-29
- Scope: sensor model (D-019); Fibonacci cap candidates with standoff R(u) = h(u) + 150 mm and scan-pass primitive (D-020); UR5 IK reachability mask over 5 sweep samples (D-021); movej PTP time matrix (D-022); coverage denominator over reachable candidates (D-023).
- Tests: `tests/test_candidates.py` (counts, unit vectors, elevation limit, sweep ⊥ view, sweep covers part, DOF near limit, determinism); `tests/test_robot.py` — IK solution exists for a known reachable pose; PTP time matrix has zero diagonal and is non-negative; IK solutions reproduce the sensor poses; movej formula; determinism.
- Files to change: src/nbv/candidates.py, src/nbv/robot.py, scripts/reachability.py, configs/sensor_v1.yaml, configs/candidates_v1.yaml, configs/robot_v1.yaml, tests/test_candidates.py, tests/test_robot.py, results/reachability_v1.csv
- Commands: `uv run python scripts/reachability.py`; `uv run pytest -q`
- Acceptance criteria: test passes; reachable-candidate ratio is reported. → Met: results/reachability_v1.csv (48 and 128 candidates, all 70 variants); plans cached in data/cache/robot/.

### Slice 3 — Visibility ✅ completed 2026-10-30
- Scope: dual-occlusion ray casting, cache; cube test; timing for one part × 48 passes.
- Files to change: src/nbv/visibility.py, src/nbv/candidates.py (explicit directions, cache load), scripts/visibility.py, configs/visibility_v1.yaml, tests/test_visibility.py
- Commands: `uv run pytest -q`
- Acceptance criteria: `tests/test_visibility.py` passes; precomputation time is reported.
- Also: fill `observable_reachable_over_all` in results/reachability_v1.csv (D-023), and the same ratio for each base position in results/base_sweep/ (train parts only).
- Result: visibility per variant ≈ 0.3 s (48 passes) / 0.5 s (128 passes), BVH build included; all 70 variants × (48 + 128) plus reports in 94 s (data/cache/visibility/timing.json). No decimation needed (1 M occlusion rays ≈ 0.02 s on the 2 M-triangle variant). Coverage columns filled in results/reachability_v1.csv and results/base_sweep/; the all-pass union is only ~41% of samples (train + val mean 40.6% at 48 candidates; D-025 finding).

### Decision point after Slice 3 — robot base position ✅ decided 2026-10-30: −350 mm (D-027)
- The base stays at (−400, 0, 0) mm until this point (D-021).
- Inputs: results/base_sweep/base_x{-350,-400,-450,-500}.csv (train parts, 48 candidates; plans cached in data/cache/robot/base_x*/), with `observable_reachable_over_all` filled (Slice 3 done).
- Rule: choose the base by coverage (reachable union / all union) on train parts and, once Slice 4 exists, cycle time; not by raw reachability (base placement in the literature is chosen by coverage / cycle time, e.g. Spensieri et al. 2025, spensieri2025). Record the choice as a DECISIONS entry; a change bumps ENV_VERSION.

### Slice 4 — Environment ✅ completed 2026-10-31
- Scope: Gymnasium environment, coverage, reward, masking, `ENV_VERSION` (fills PROTOCOL.md → ENV_VERSION); `tests/test_env.py`, `tests/test_determinism.py`.
- Option (default off): randomize the part's yaw on the table and a small position offset per episode, for Phase 2 data augmentation (see Plan 003 → Open issues).
- Joint-state tracking (D-024): transfer time per joint from the [−π, π] difference, with a ±360° limit check over the whole next pass; the arm's state after a pass is its (shifted) end configuration.
- Information model (D-029): passes and IK from the nominal mesh per base part; nominal and real visibility with the same passes; observation without privileged data; features in src/nbv/features.py.
- Files to change: src/nbv/env.py, src/nbv/features.py, src/nbv/robot.py (transfer), scripts/reachability.py, scripts/visibility.py, scripts/env_benchmark.py, configs/env_v1.yaml, tests/test_env.py, tests/test_determinism.py
- Commands: `uv run pytest -q`
- Acceptance criteria: tests pass; episode duration is measured (input for the step budgets); cycle time is computed by joint-state tracking and a test checks it against a hand-computed sequence (D-024).
- Result: random policy ~5 100 steps/s (48 candidates) and ~6 100 steps/s (128), ~4.5 ms per episode (scripts/env_benchmark.py). With nominal plans, reachability is 0.81 (48) / 0.80 (128) on train + val parts; with the v1 deviations the real and nominal reachable unions differed by 2.4% on average (max 5.3%; train + val, 48 candidates). A greedy plan made on the nominal visibility to 95% nominal coverage reaches 95% real coverage on 33 of 40 train variants (min 93.2%): small deviations matter at the threshold. With v2 deviations (D-031): real/nominal reachable unions differ by 1.1% on average (max 3.7%; train + val, 48 candidates); the same plan reaches 95% real on 35 of 40 train variants without repair (min 93.8%), and on 40 of 40 with plan repair (D-032; 1–2 repair passes on 5 variants).

### Slice 5 — Baselines ✅ completed 2026-10-31
- Information per baseline (D-029): greedy-IG on nominal visibility; greedy-IG/cost on nominal visibility and ptp_pi; greedy-oracle on the real variant's visibility (upper reference); set-cover ILP + TSP solved on the nominal CAD and evaluated on the variant through `nbv.env.evaluate_sequence`.
- Scope: random, random + TSP (K random reachable passes ordered by TSP; competitive at large budgets in ObjView-Bench), fixed pattern, greedy variants, greedy-oracle, ILP + TSP; runs on validation parts, registry rows.
- Per-part planning difficulty d_plan: minimum number of passes covering the reachable union (set-cover ILP, 48 candidates); report the split's balance in d_plan alongside the hard-to-see ratio.
- Tests: `tests/test_baselines.py` — consistency: at the same number of passes k, greedy-oracle coverage (real visibility) ≥ greedy-IG coverage, for every k. Not for cycle time: both are myopic, so neither is guaranteed to be faster.
- Files to change: src/nbv/baselines.py, src/nbv/evaluation.py, src/nbv/env.py (evaluate_policy, joint motion), scripts/run_baselines.py, scripts/d_plan.py, scripts/ilp_bound.py, tests/test_baselines.py, tests/test_evaluation.py, results/registry.csv, results/d_plan_v1.csv, results/ilp_bound_v1.csv
- Commands: `uv run python scripts/d_plan.py`; `uv run python scripts/run_baselines.py`; `uv run python scripts/ilp_bound.py`; `uv run pytest -q`
- Acceptance criteria: test passes; all baselines have rows in `results/registry.csv`; every baseline's pass sequence (including the TSP tour) is re-evaluated in the environment and only those cycle times are reported (D-024).
- Result: 140 exploratory registry rows (8 methods × 48/128 × train/val; stochastic methods seeds 0–9); outputs/p001s5-*/. All 56 set-cover and 56 max-coverage ILPs solved to optimality. Every method stays below the k-pass ILP bound on 2 train + 2 val variants (k ≤ 14); first as scripts/ilp_bound.py → results/ilp_bound_v1.csv, since Slice 6 a test in tests/test_baselines.py (script and CSV removed).
- Addendum (D-035, D-036): joint-optimum reference (scripts/joint_opt.py, results/joint_opt_v1.csv), difficulty-normalized cycle time (t / t_ILP+TSP on the same variant), K_max 40 at 128 candidates (ENV_VERSION v2). Set cover made reproducible (two-stage CP-SAT, D-033). Registry: 144 active exploratory v2 rows (9 methods); 428 earlier rows superseded (v1 run; two v2 runs with irreproducible ILP covers).
- Finding (d_plan): the split is balanced in hard-to-see ratio by construction, but not in planning difficulty. With the ENV v4 masks (results/d_plan_v2.csv) the mean d_plan95 at 48 candidates is 9.8 (train), 11.7 (val), 12.3 (test); the imbalance is smaller than with the v1–v3 masks (10.6 / 12.3 / 13.7, results/d_plan_v1.csv, historical) but remains. Reported as a limitation; test values are not used for any decision.

### Slice 6 — Protocol freeze ✅ completed 2026-11-03
- Scope: fill the remaining `<...>` fields in docs/PROTOCOL.md; freeze it with a DECISIONS entry.
- Files to change: docs/PROTOCOL.md, docs/DECISIONS.md
- Acceptance criteria: PROTOCOL.md `Status: FROZEN`; DECISIONS entry added.
- 6a (2026-11-01): Phase 2 dependencies (torch cu128, sb3-contrib) and a MaskablePPO speed trial (scripts/phase2_speed_trial.py; throughput notes in Plan 003).
- 6b (2026-11-02): sanity check before the freeze (docs/reviews/2026-11-02-sanity-check.md); verdict: not freezable as it was.
- 6c (2026-11-03), fixes from the sanity check: run provenance (D-037); ENV v3 with objective-only reward, agent-side shaping, K_max as a terminal state and the agent/eval info split (D-038; baseline metrics verified unchanged on 2 970 episodes); statistics wording and per-part presentation, scripts/report.py (D-039); PROTOCOL v1 scope (D-040); canonical baseline set (D-041); incidence sensitivity at the visibility level (results/incidence_sensitivity_v1.csv); collision diagnostic (D-042, results/collision_v1.csv); cleanup (dead caches, raw PTP matrices, env_v1, ILP bound script → test).
- 6d (2026-11-03): collision check in the masks (D-042, 5 mm margin, ENV v4, configs/robot_v2.yaml): reachable passes on train + val parts 427 → 374 (48 candidates) and 1 126 → 996 (128); measurable surface share 0.366 → 0.351 (48). Base position re-checked by the predeclared rule (D-027, results/base_check_v2.csv): −350 mm stays. Regenerated: robot plans and masks, results/reachability_v2.csv (coverage columns from the cached visibility, which does not depend on the masks; checked bit for bit on 4 meshes), results/d_plan_v2.csv, joint-optimum plans and results/joint_opt_v2.csv (48 candidates), results/incidence_sensitivity_v2.csv; canonical baseline set rerun as exploratory v4 rows (run_ids p001s6x-*). No v3 rows existed to supersede. results/figures/ moved to scratch/figures/ (temporary trials).
- 6e (2026-11-03): second sanity check (docs/reviews/2026-11-03-sanity-check-2.md) and its pre-freeze fixes (collision evidence protected, robot cell numbers and censoring in PROTOCOL, cache and code hashes in provenance); PROTOCOL v1 frozen (D-043); canonical baseline runs (results/baselines_v1.md); literature status levels (D-044); Plan 002 skeleton.
- Acceptance criteria met: PROTOCOL.md `Status: FROZEN`; D-043 added.

## 4. Open issues
- Self-intersection is not checked (D-018 checks only watertight, winding, volume). Displacements ≤ 1.5 mm with normal-weighted dents make it unlikely except on walls thinner than ~3 mm; revisit if Slice 3 shows ray-casting artifacts.
- The hard-to-see ratio used for stratification is a screening proxy (orthographic, camera-only, 2000 samples); it is not a protocol metric.
- `thingi10k` must be imported before open3d in a process (lagrange DLL conflict on Windows); only `nbv.parts.fetch_parts` imports it, lazily.
- UR analytic IK (up to 8 closed-form solutions) is an option to evaluate later: exact branch enumeration instead of seeded LM (D-021).
- Cycle time excludes data processing time (registration, CAD comparison) (D-036 limitation).
- Turntable variant: the cell has no turntable, unlike common industrial robot + turntable scanning cells (e.g. ZEISS ScanCobot); most robot-side unreachability comes from this (D-027 limitation). A turntable (extra joint, or a set of part poses via `pose_id`) is a candidate extension.
- Robot–part and robot–table collisions are checked statically at the 5 IK samples of each pass since ENV v4 (D-042); robot self-collision, transfers between passes and fixtures are not checked.
- Quality-weighted coverage (future work): weight each measured sample by its expected precision (incidence angle, position on the laser line) instead of the binary model of D-025.
- Sweep direction is fixed (always start → end along +sweep); the reverse direction of the same pass is out of scope.
- Dual-camera sensor (one camera on each side of the laser) as a possible ablation; Slice 3 stores laser and camera visibility separately so a second camera can be added without changing the matrix layout (`camera_sides: [1, -1]` in configs/visibility_v1.yaml).
- "Geometry unknown" (non-model-based NBV) is out of scope: it does not match the quality-control problem (D-029). Possible future work for reverse engineering or legacy parts without CAD.
- ISO 8062 check: v2 deviations (D-031: scale ±0.75% ≈ ±1.5 mm at 200 mm, noise ≤ 0.5 mm, dents ≤ 1 mm) still have to be compared with the CT 8–10 table of ISO 8062-3 before writing that they match these grades.
- Only ~40% of the sample surface is measurable from any pass (D-025): the part is never flipped, and hollow parts have much deep interior surface. Coverage is reported relative to the reachable union (D-023); a second setup (part flipped) is listed in Plan 003 → Open issues.
- Closed on 2026-11-03 (sanity check A3): statistics now follow D-034 / D-039 (no IQM / stratified bootstrap); torch comes from the cu128 index (Slice 6a, D-009); mesh size was no problem for RaycastingScene (Slice 3); the base is at −350 mm (D-027), so the −400 mm finding is historical.
- Disposition at closure (2026-11-03):
  - Recorded limitations, no further work planned in this project (listed in PROTOCOL v1 → Model limitations or in DECISIONS): self-intersection check, screening proxy for the split, data processing time, sweep direction fixed, geometry-unknown NBV, static collision check only (D-042), ~40% measurable surface.
  - Carried to Plan 003 → Open issues: turntable (Plan 004 / future work), second setup with the part flipped, dual-camera ablation, quality-weighted coverage.
  - Carried to Plan 002 → Open issues: ISO 8062 comparison (needed only if a report claims CT 8–10), UR analytic IK (only if IK becomes a bottleneck).
  - Implementation note kept for any session: import `thingi10k` before open3d (Windows DLL conflict).

## 5. Closure (filled 2026-11-03)
- Outcome:
  - A tested NBV environment (ENV v4): model-based information model, analytic dual-occlusion visibility on 14 Thingi10K parts × 5 cast-deviation variants (8 / 3 / 3 split by base part), UR5 cell with IK and static collision masks (base −350 mm, 5 mm margin), joint-state-tracked cycle time, objective-only reward with agent-side potential shaping; 135 tests, reproduced bit for bit from a clean clone (two sanity checks).
  - Canonical baselines at 48 candidates on train and val (results/baselines_v1.md, from registry rows only): all main-table methods reach 95% on every variant; normalized cycle time (t / t_ILP+TSP, mean of part means) on val: greedy-IG/cost 1.186, MCTS 1.104, ILP + TSP 1.000, joint optimum 0.984; on train 1.180 / 1.094 / 1.000 / 0.990. The room for a learned planner lies between the best simple methods and the solver references.
  - The joint optimum improves the industry-type ILP + TSP planner only slightly (planner objective 0–3.2%, results/joint_opt_v2.csv); in the environment the advantage is 1–2% and part-dependent.
  - Limitations recorded in PROTOCOL v1: ideal binary sensor, single setup without turntable (~40% of the surface measurable), static collision check only at the 5 sweep samples, split imbalance in planning difficulty (d_plan95 train 9.8 / val 11.7 / test 12.3).
- Registry rows added (run_ids): 50 canonical rows `p001s6-<method>-n48-<split>-s<seed>-v4-20261103T170747-<6 hex>` (greedy_cost, mcts ×10 seeds, ilp_tsp, joint_opt, random ×10 seeds, fixed, greedy_ig; train and val), commit b8240eb, clean tree. All earlier rows are superseded: 140 v1 and 432 v2 rows (Slice 5, exploratory), 50 v4 rows (`p001s6x-*`, exploratory).
- Decisions added: D-001…D-044. Environment and cell: D-003…D-006, D-011, D-012, D-019…D-030, D-036, D-038, D-042; parts and split: D-015…D-018, D-031; information model: D-029; baselines and statistics: D-032…D-035, D-039, D-041; protocol and process: D-001, D-007…D-010, D-014, D-037, D-040, D-043 (freeze), D-044 (literature status). Superseded: D-002 (by D-029), D-010 (by D-034), D-013 (by D-016), D-018 (by D-031).
- Docs updated: docs/PROTOCOL.md (frozen v1), docs/DECISIONS.md, docs/LITERATURE.md, docs/PROJECT.md (Slice 1), README.md, project status notes (current phase, policy-code location rule), docs/plans/003-deep-rl-and-ablations.md (Open issues), docs/plans/002-classical-rl.md (skeleton), docs/reviews/2026-11-02-sanity-check.md, docs/reviews/2026-11-03-sanity-check-2.md.
