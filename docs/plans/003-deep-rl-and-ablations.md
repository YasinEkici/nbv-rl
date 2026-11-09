# Plan 003: Phase 2 — deep RL and ablations

Status: draft
Opened: 2026-10-29   Closed: —
Related decisions: D-008

Placeholder created during Plan 001 to collect items for Phase 2. Sections 1–3 are written when this plan is drafted (docs/plans/TEMPLATE.md).

## 1. Goal
Draft question: can learned state-action interactions improve planning beyond Phase 1's fixed linear pass score while preserving the environment, reward and comparable metrics? The user must approve the actual design, PROTOCOL v2 and budgets before implementation or training.

## 2. Predeclaration (fill BEFORE any result is seen)
- Runs: <...>
- Seeds / budget:
- Comparisons to report and test used:
- This plan will NOT:

## 3. Slices
<...>

## 4. Open issues
- Carried from completed Plan 004 (D-070): exported VAL models 271875, 591247 and 37275 have no recorded license version. Resolve their source versions before public redistribution; never infer a deed. Keep attribution/changes/share-alike and derived-coverage labels in any future viewer. The live demo has fixed-case headless checks and one interactive browser inspection, not exhaustive browser/seed coverage; runtime exact-match warnings must remain visible. The original Phase 1 test set stays closed.
- Options, not decisions (carried at Plan 002 closure, D-067): dueling masked DQN with value/advantage factorization (wang2016_dueling); HuRL-style shorter-horizon guidance using value_estimate (cheng2021_hurl); residual RL over a greedy or MCTS base policy (silver2018_rpl; discrete masked actions need an explicit adaptation). No option is selected or implemented here.
- Test the unverified explanation that a useful potential reduces the state-value burden and helps action distinctions (D-065). Predeclare controls using train/VAL and an appropriate future holdout policy; the closed Phase 1 test must not guide adaptive selection.
- Carried Phase 1 regularization limitation: empirical RegQ stability does not establish a sufficient eta or a theorem guarantee (D-058). The amended eta grid does not stabilize every constructed divergent example; any reused learner needs finite/norm monitoring and predeclared handling.
- Carried constant-step limitation: tail averaging reduces last-weight fluctuations, not all asymptotic bias. State explicitly whether/how future algorithms average weights; preserve final-policy secondary reporting where comparable.
- Carried information-boundary rule: all agent/policy code stays under src/nbv/agents/; static/runtime privileged-data checks must cover new encoders and policies. Scripts are thin orchestration, not a place to bypass the checks.
- Preserve legacy feature-check provenance and the invalid diagnostic-summary notice (D-059), rather than rewriting original outputs/rows. Slice 3's single learning loop and eligible-cache hashing resolved the earlier infrastructure items.
- Early policy/repair None censoring is already tested (D-014); ensure future continuous/discrete action adapters preserve the same terminal/censored semantics. Shaping choices remain agent settings; none/linear equivalence is representation-dependent.
- Conditional source/engineering debts carried from Plan 002: verify ISO 8062 CT 8–10 before claiming those casting grades; consider analytic UR IK only if profiling makes seeded LM a bottleneck. Do not revise geometry or the split to improve results. Gap comparisons must use the same-part reference, not the historical VAL aggregate.
- Future test evaluation must store learner per-step traces, including selected pass, coverage, motion/scan/home times and cumulative state, before opening the holdout (D-064). Plan 002 independent validation checked recorded consistency, not learner-policy replay; its closed test set will not be reopened to repair missing traces.
- Mandatory before training-infrastructure reuse (D-060): publish divergence from live workers and stop the parent when >5 seeds of an arm diverge; the current counter runs only after full tasks complete. Plan 002's 70 runs had no divergence. Deferred explicitly during Slice 5b; fix and test before any new training.
- PROTOCOL v2 correctness list: add tests/test_agents.py and tests/test_features.py (Plan 002); frozen PROTOCOL v1 is left unchanged.
- Out-of-distribution test candidates (noted in Plan 001 Slice 1, not downloaded): Thingi10K file_ids 741528 (Hyperbolic Planetary Gearset), 669967 (Build Your Own Jet Engine), 40900 (Twisted Bottle Whistle), 42638 (LED mounts), 67223 (Solder Holder); all CC BY / CC BY-SA, watertight. Candidate data and screening metrics: scratch/thingi10k_candidates/ (not committed; rerun the Slice 1 screening if lost).
- Throughput notes from the Plan 001 Slice 6 speed trial (scripts/phase2_speed_trial.py, MaskablePPO, 48 candidates): (1) set OPENBLAS/OMP/MKL_NUM_THREADS=1 before numpy is imported in env workers, otherwise every SubprocVecEnv worker commits ~1.8 GB (OpenBLAS buffers for 24 cores) and 16 workers run out of memory; then set torch.set_num_threads explicitly in the learner. (2) Import torch only in the learner process (CUDA torch in every worker also costs a lot of memory on Windows). (3) The learner, not the environment, is the bottleneck: 16 envs were not faster than 8. The MLP sees a 20 000-bit observation; the 3D CNN run used only ~15% GPU, so Python voxelization and host-to-device copies dominate. Vectorized voxelization and a compact observation are the first speedups to try.
- Scan speed sensitivity (100 and 300 mm/s): industrial profilometers reach profile rates in the kHz range; faster scanning raises the share of motion time and makes the ordering matter more.
- Phase 2 arms: SAC (the method suggested by the assignment, with a continuous (Δφ, Δθ) action) is one of the two main arms, not a side experiment; the other is masked PPO / DQN over the discrete candidates. Final decision in this plan's predeclaration (refines D-008).
- PROTOCOL v2 (D-040): define at the start of this plan, before any Phase 2 result: SAC's continuous action and its coverage reference, which baselines run at 128 candidates, the Phase 2 budget (input: the speed trial above).
- SAC action, one option: map the continuous action to the nearest reachable candidate of a dense candidate set (Dulac-Arnold et al. 2015, "Wolpertinger"; unverified).
- Second setup (Plan 003 / 005, descriptive only): flipping the part for a second setup is common in industry (e.g. the Laser Design flip-plate fixture) and would open the table-facing samples (2–20% of the samples per train/val part, normal within 10° of −z; D-025). Compute the measurable surface share with two setups descriptively, without changing the main problem.
- Turntable (Plan 005 / future work): common in industry (ZEISS ScanCobot, Shining 3D RobotScan, Kwon, Na & Song 2019); not added in this project because it would mean redoing most of Plan 001 (D-027).
- Carried from Plan 001 (2026-11-03): dual-camera sensor ablation (layout ready: `camera_sides` in configs/visibility_v1.yaml); quality-weighted coverage (weight samples by expected precision instead of the binary model of D-025).
- Data augmentation: Plan 001 Slice 4 adds an env option (default off) that randomizes the part's yaw on the table and a small position offset. Decide here whether Phase 2 training uses it; evaluation keeps it off unless predeclared.

- ENV v5 proposal from D-071 (not implemented): retract intermediate poses and densely sampled collision checking for start/inter-pass/home transfers and scan interpolation; recompute every affected transition-time matrix. This changes the motion/environment model and requires regenerated baseline/learner results and a new protocol, not retrospective corrections to frozen Phase 1 numbers. See results/collision_audit_v1.md.

- Plan 004b closure: source license versions for VAL demo parts 271875, 591247 and 37275 remain unrecorded; retain version-unknown notices until source verification. Optional final-weight train collision measurements were omitted, not pooled with primary results.

## 5. Closure (fill when closing)
- Outcome (2–5 bullets):
- Registry rows added (run_ids):
- Decisions added:
- Docs updated:
