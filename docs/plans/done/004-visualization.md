# Plan 004: Visualization and demos

Status: completed - S0-S4 delivered under D-068 through D-070
Opened: 2026-11-08   Closed: 2026-11-08
Related decisions: D-063, D-064, D-066, D-067, D-068, D-069, D-070

S0, S1/S2 (D-069), and S3/S4 (D-070) were approved and completed on 2026-11-08. Earlier S0-only/pending statements describe their historical checkpoint. Plan 003 stays a draft awaiting separate approval and PROTOCOL v2; final analysis remains Plan 005.

## 1. Goal

Explain the environment and existing results for the 13 November 2026 midterm, README and a later paper. No new experiment, tuning, selection, learner, hypothesis or changed result. The external viewer will be built externally from the S2 files; possible later repository hosting is docs/demo/.

## 2. Predeclaration

- Every number comes from results/registry.csv or identified stored outputs/caches, not manual transcription. Figures/tables must be reproducible with source run_ids, snapshot/manifest references, units and transformations. Core logic belongs under src/nbv/; scripts are thin entry points.
- The Phase 1 test holdout is closed (D-063/D-064). Show test results from **registry metrics only**. Never construct a real test environment, read test caches, read test episode outputs to fill registry gaps, or replay test episodes. Synthetic fixtures are not the real holdout.
- Stepwise visuals and demos use VAL/train only. Replay stored pass sequences rather than make new policy choices. Recomputed sequence, coverage, times and joints must be checked against original records; sequence, cycle time, success and recorded final coverage must match exactly. A mismatch blocks publication. Preserve a verification outcome with each export.
- If a required sequence is absent, policy-based reconstruction is exceptionally VAL only, original fixed seed-0 checkpoint, exploratory, with exact reproduction of recorded results. This is not authorized in S0. Missing independent comparison fields are not permission to invent targets or relax equality; see Open issues.
- Reconstruct physics with the original nominal model and caches. Mesh simplification is display-only and must not change visibility, masks or metrics. Distinguish nominal and deformed-variant sample coordinates and retain their shared face/barycentric correspondence.
- Every published mesh/derivative visibly carries the Thingi10K/Thingiverse source, author and CC BY / CC BY-SA license in the bundle and viewer. Share BY-SA derivatives under the same license; keep dataset licensing separate from software licensing.
- Use colorblind-friendly colors and redundant shapes/line styles. State what each band/interval means: bootstrap interval, seed min–max or percentile. No new significance tests; the base part remains the generalization unit.
- Decision latency is D-066's VAL, seed-0, this-machine measurement of warm-cache calls. Exclude environment/model preprocessing as in that record. Combining it with test quality is explicitly a cross-split descriptive view, not matched test timing.
- Preserve PROTOCOL v1, ENV v4, settings, weights, original outputs, registry quality rows, test claims and hypothesis decisions. The earlier S0 restriction was documentation-only; D-069 then authorized S1/S2 code, rendering and VAL derivative export. No installation or dependency/lockfile changes were needed.
- D-070 subsequently authorizes S3 live inference in configured VAL v0–v4 only, with frozen methods/settings and original-source exact checks. This exception does not permit a new experiment, training/selection, test use or output/registry/bundle modification. Viser is opt-in, with a separate dependency/lock commit. The S1/S2 bundle remains immutable.

## 3. Slices and acceptance

### S0 — Plan and read-only inventory (this turn)

Check numbering and live references; append D-068 with table row and detail. Inventory registered metrics, VAL episode fields, seed histories, allowed visibility/point/robot caches, mesh capability/sizes and non-installing Viser compatibility. Document missing data and gates. Observe missing-document acceptance checks fail before writing and pass afterwards; run full existing pytest before commits. There is no new behavior requiring an artificial implementation test in this documentation-only slice.

### S1 — Static figures (completed)

Implement scripts/report_figures.py or extend existing reporting; export PNG + PDF with reproducible data/source metadata. Write meaningful synthetic tests first and observe red before implementation. Plotting paths may not open real test environments/caches/episode files.

| Item | View | Data and acceptance |
|---|---|---|
| a | Quality–decision-time/Pareto view | Five D-066 policies; x = ms/decision (log), y = registered test normalized cycle time; explicit VAL/seed-0/hardware timing note. No inferred timing or deployment-optimality claim. |
| b | Best/lower-median/worst VAL runs | Ten faint original curves, rank by registered official tail-policy VAL score then seed. Highlight sorted indices 0/4/9 as actual runs; no percentile labels. Store alternative whole-curve-mean order as information only (D-069). |
| c | Per-part test dot plots | Separate panels with registry variant × seed points; respect deterministic baselines' actual counts. Label descriptive/paired nature and interval meaning. |
| c2 | Paired H-forward differences | D-055 paired_difference on registry variant means; separate part panels, zero/-0.02 guides and split mean; match the original full-precision verdict and its .15f display exactly. |
| d | Coverage-time curves | VAL 271875_v0, seven original seed-0 methods. Baseline saved steps; learners from full-oracle-validated sequence replay plus exact partial summary checks; explicit home segment and derived-final-coverage label. |
| e | VAL time decomposition plus test joint-motion/time points | VAL: move/scan/home for fifteen variants, all six learner arms/ten seeds and four baselines. Test: registry-only joint-motion x versus normalized-time y, distinct part markers; no test time components (D-069). |
| f | VAL/train part gallery | Per-sample number of reachable passes that measure it; table-facing and never-seen samples separately labeled/colored. Document orientation criterion; use existing visibility, not new ray tracing. |
| g | Candidate azimuth/elevation map | All 48 directions per allowed part, separating reachable, IK-unreachable and collision-masked using original masks; 2D degree axes/frame explicit (D-070). Legacy sphere filenames retained. |

### S2 — Demo data export (completed)

Export **271875_v0, 591247_v0, 37275_v0 (VAL)** only:

- Normalized nominal display mesh, approximately 50000 triangles maximum, with source and simplification parameters. Raw nominal meshes already meet the target; avoid exporting huge subdivided/variant meshes unnecessarily.
- Surface sample positions/normals and variant measured pass × sample packed bits. Preserve sample order and declare frame, mm units and nominal/variant distinction; any display-only conversion is separate from full-precision physics.
- All 48 sensor start/end positions, view and sweep directions, reachability and reason masks, derived from saved pass geometry rather than new IK.
- Robot DH parameters (library/version provenance), base transform, TCP, q_home, five IK samples per pass, capsule radii/offsets and sensor box. Unreachable q entries may be invalid: retain explicit validity/masks and binary typing, not invalid JSON numeric literals; never animate invalid passes.
- Seven episodes per part: greedy_cost, MCTS, ILP+TSP, joint_opt, SARSA/linear, SARSA/value_estimate and RegQ/linear. Learners/MCTS use original seed 0 and primary tail policies. Deterministic baseline registry seed cells are empty although their run/config identity is s0; use that identity rather than select another run.
- Step rows: pass, move time, scan time, cumulative time, primary coverage and derived joints; a final explicit return-home row. D-069 resolves the former gate: full recorded baseline steps validate the tool, learners match every stored summary. Learner final coverage/joints remain explicitly derived and are not claimed independently recorded; matching AUC is not a final-coverage proof.
- Versioned JSON schema with dtype, shape, byte order, base64 encoding, bit order/count and units. Tests cover schema, exact matching, finite JSON scalars, validity/lengths, attribution and total bundle size **<=12000000 bytes** (decimal 12 MB), including JSON/base64 overhead and SHA-256 manifest. Hash every exported file. No silent precision loss to fit the cap.
- Output results/demo/ is the user-authorized derivative-data exception for this plan, with attribution/license review; raw data/cache/ and outputs/ stay uncommitted. Viewer implementation and possible docs/demo/ hosting are later work, not S0.

### S3 — Optional local live demo (completed)

VAL only, Viser, separate optional dependency group. Verify Python 3.14 installation/startup first; drop S3 if unavailable. S0's binary-only dependency dry run succeeded for Viser 1.1.1 but did not install/test a server or browser. Rerun and marimo are software references only if actually used; neither is adopted here.

D-070 setup/startup passed inside the 30-minute box; S3 was not dropped. `uv sync --extra demo` installs the opt-in group. `scripts/live_demo.py` / `src/nbv/visualization/live.py` run local VAL inference or SHA-verified recorded-bundle playback. Learner inference remains under agents/frozen.py. Core checks reject non-VAL IDs before geometry/env access, validate manifest-bound weights and compare stored outcomes exactly. CAD plans are replayed, MCTS preserves its canonical cross-variant RNG stream. UI interpolation is display-only, timings are not D-066-comparable. See the guide and validation artifacts listed in Closure.

### S4 — Closure (completed)

Record figures/demo manifests, reproducibility checks, unresolved gates and licenses; archive/update project status and README, locally fast-forward/delete the local branch, never push. Do not activate deep RL as a side effect.

## 4. Schedule and estimates

S0: 8 November. S1 + S2: 8–9 November after approval and resolution of gates. External viewer: 9–10 November. S3 only if time remains. Midterm: 13 November 2026.

Engineering estimates, not measured results: S1 4–6 hours; S2 5–8 hours, plus review/rework for missing references. Optional S3 is excluded. The 12 MB cap is tested only on actual S2 serialized output; S0 contains estimates, not a completed export.

## 5. Closure

- Static figures were delivered and corrected under D-070: unique small-figure colors/family encodings, fixed overlap order, paired mean lines, actual-range cividis galleries and 2D candidate maps. Original scientific values and old min–max figures remain. [Generated figure/delivery index](../../../results/visualization/README.md).
- The immutable [VAL bundle](../../../results/demo/README.md), schema and manifest remain unchanged from S2. Exact full/partial sequence replay and labels are preserved; the local recorded player verifies SHA-256 before reading its arrays. No test cache/output/environment was opened.
- The optional live VAL viewer completed; headless checks for each live method and stored CAD plans match original sequences and all available summaries exactly. [Diagnostic](../../../results/visualization/live_validation_v1.json), [actual Chromium screenshot](../../../results/visualization/live_demo_v1.png) and [capture provenance](../../../results/visualization/live_demo_v1.json). No automated browser-test suite was introduced; GUI inspection completed a SARSA/linear episode through home with the exact-match label visible.
- The allowed mask audit found a differing train/VAL pair, disproving constant masks over all fourteen without opening test caches. Directions share a Fibonacci set, sensor positions depend on the part, and existing VAL diagnostics include both part and table collision limits. [Audit](../../../results/visualization/candidate_audit_v1.md). It does not claim per-test geometry checks or a universal table-only cause.
- Registry rows added: **none**. Training, tuning, methods, selected settings, reward/model, protocol, test results and demo bytes/hashes remain unchanged. Decisions added: D-068 opening/numbering, D-069 S1/S2 oracles/coordinates, D-070 corrections/opt-in live viewer/closure. Full pytest precedes every commit using the required failure-stopping chain. The full optional-runtime suite and loopback smoke pass; read-only integrity checks preserve frozen scientific inputs.

Outputs and reproduction: scripts/report_figures.py, scripts/report_candidate_masks.py, scripts/report_visualization.py; scripts/live_demo.py (`--recorded`, live or `--validate-headless`), [user guide](../../DEMO_GUIDE.md). Demo export/replay regeneration remains a controlled fresh-output operation, not a playback prerequisite. Updated docs: README, project status, PROJECT, DECISIONS, LITERATURE, this archive, S0 inventory links and draft Plan 003's carried issues. Viser's optional pyproject/lock change was committed separately.

Open issues carried with reasons to [Plan 003](../003-deep-rl-and-ablations.md): missing model license versions before public redistribution; retaining attribution/derived-field notices; fixed-case rather than exhaustive browser/seed coverage. Runtime checks remain mandatory for any selected live episode. S3 was not dropped. There is no permission to reopen Phase 1 test or start Phase 2. Close the branch by local fast-forward into main and delete it; never push.

## 6. Sources

docs/LITERATURE.md records checked sources/status: tanaka2026 (run-wise percentile highlighting), agarwal2021 (interval estimates/performance profiles; no change to PROTOCOL v1 statistics), yi2025_viser (web-based Python visualization), all verified-source. rliable has been archived/read-only since 15 October 2025; do not add it. Rerun/marimo remain unselected software options.

## 7. Open issues and resolved gates

- Resolved by D-069: no test time decomposition; VAL components and registry-only test motion/time view instead.
- Resolved by D-069: baseline full-step oracle validates replay; learners use every stored partial summary exactly. AUC is one consistency scalar, not a mathematical final-coverage oracle. Label derived learner final coverage accordingly.
- Future allowed reconstruction must preserve ENV v4's joint-wrap/limit behavior; q_plan alone does not establish the executed lifted state. No relaxed tolerance or visually preferable sequence may silently replace a mismatch.
- Resolved by D-069: sample coordinates/normals are variant v0 only; display mesh nominal. Keep correspondence order and explicit coordinate/deviation labels.
- Resolved by D-070: Viser was installed as an optional extra and both bare/application server and actual browser startup worked; fixed-case headless and loopback smoke tests pass. Browser/seed coverage is not exhaustive; preserve runtime matching and explicit warnings.
- License versions for VAL parts 271875, 591247 and 37275 are missing from source split metadata. Bundle attribution says "version not recorded", specific deed URLs are null, CC overview is labeled as such. Resolve original versions before public redistribution; do not invent a version. The external viewer must visibly show attribution, family/status and modifications.

## 8. Historical slice outcomes

S0 completed on 2026-11-08. Detailed sources, array/file sizes, episode-field gaps, successful Viser dry resolution and estimates: [S0 inventory](../../inventories/004-visualization-s0.md). At that handoff S1–S4 had not started and its two data gates awaited approval. D-069 resolved them; S0 itself created no metric row, environment, figure or demo.

S1/S2 completed on 2026-11-08. [Generated delivery](../../../results/visualization/README.md) contains oracle counts/matched fields, exact paired difference, figure links, alternative learning-curve rankings, real serialized bytes, complete external-viewer upload list and unresolved license versions. Sources are the replay/figure/export manifests, not manual numeric transcription. Existing min–max figures remain; the README encoding typo was corrected and Markdown scanned as UTF-8. S1/S2 introduced no policy evaluation, selection, test cache/output/environment access or changed registry row. D-070 later authorized the separately labeled live VAL demo.

Reproduce with scripts/replay_visualization.py (preserved exclusive original VAL validation record), scripts/report_figures.py, scripts/export_demo.py (fresh destination), and scripts/report_visualization.py. Core implementation is src/nbv/visualization/. Regression tests cover exact full/partial oracles, corrupted records, holdout I/O boundaries, palette/rank/paired arithmetic, lossless schema/bit order, finite/masked joints, licensing, byte cap/hash tampering and Windows-safe publication. The demo bundle disables Git text conversion so its byte hashes survive Windows checkouts. S3/S4 are completed by D-070; stop before any next-plan work.

## 9. Presentation correction addendum (2026-11-10)

[Design correction closure](004-design.md), D-072: genuine initialized-session startup is fixed at the joint-state boundary; theme v2 replaces subset-dependent styles with a measured global mapping. [Current gallery](../../../results/visualization/design_v2.md) indexes versioned English reports, Turkish gray/highlight slides, portable fonts and new browser captures. Original data, bundle styles, figures, evidence and manifests remain intact. [Historical link exceptions](../../../results/provenance/link_exceptions_v1.json) narrowly retain four old D-069 link errors. The S0 inventory is now tracked. Source license versions remain unresolved; the GUI uses its documented default font because no supported local-font API is exposed. No scientific result or holdout status changes.
