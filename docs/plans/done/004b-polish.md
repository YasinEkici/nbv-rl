# Plan 004b: Collision diagnosis and presentation polish

Status: completed. Opened: 2026-11-09. Closed: 2026-11-10.
Branch: plan/004b-polish, from main. Project document; private workflow configuration remains excluded.

## Scope and ordering

A measures collisions, never corrects trajectories. Source: the 195 full-oracle baseline and 900 partial-oracle learner VAL traces, then canonical train primary tail evaluation sequences plus their canonical baseline counterparts. Secondary final-weight train records, if reported, remain a separate appendix rather than pooled with primary policies. No policy, training, solver, test cache/output, quality-registry write or frozen demo export. Geometry is the original normalized/subdivided nominal CAD used by D-042; this is an audit of the existing nominal collision model, not a new real-variant robustness model.

Use unchanged collision.pose_clearance: link capsules, sensor-head box, part/table distances, sampling/table exemptions and near_mm. Check original five IK samples separately and STOP on any negative clearance there; do not relax tolerances or fix the path. Then sample each executed joint-space transfer, four in-scan subsegments, and return home at maximum 1 degree per joint between adjacent samples. Preserve executed multi-turn angles. Report geometric path fractions; display-time fractions assume the viewer's linear interpolation and are not a certified trapezoidal robot trajectory. Exact repeated paths may share a geometry computation; retain every episode's denominator/event identity.

Deliver results/collision_audit_v1.json/.md with source identities, completeness, sampling, per-part/method denominators, rates, penetration distributions and worst events. Screenshot the worst three in local Viser. A4 adds only a generated limitation sentence/link to README/test prose; numeric tables/verdicts, registry and immutable bundle hashes remain unchanged. D-071 records scope/measurement and future ENV v5 work, not a present environment change.

B is conditional on A's waypoint gate. Verify a licensed, offline-available UR5 visual description against the existing DH flange/TCP at every saved valid IK sample before enabling it; <=0.5 mm and <=0.1 degree. If not matched, retain capsules and report. Keep point meanings, show the collision model separately, preserve unsafe paths and expose audited collision intervals. Add comparison playback on a common simulated clock, camera presets, tabbed controls/chart/coverage indicator. Existing report figures remain byte-identical; new 16:9 slide exports live under results/figures/slides/. No result/setting/hypothesis changes. Literature rougier2014 is source-checked before verified-source status.

## Acceptance

- Observe red before new behavior, then full pytest before each commit (`uv run pytest -q && git commit ...`, no pytest pipe); no push.
- Sampling/endpoint STOP, split guards and exact recorded-path reconstruction tests precede the audit.
- Mandatory B regression checks: URDF/DH alignment, interval consistency with A JSON, same comparison clock; preserve loopback HTTP smoke.
- Preserve original result numbers and full immutable bundle hashes. Keep private workflow configuration local-only.

## Progress / open issues

- Input inventory confirmed original train episode sequences and immutable VAL joint traces.
- Exact train replay completed: canonical baseline and primary/secondary learner source records all pass the unchanged D-069 oracle. Keep final-weight rows separate. Audit input is scratch/p004b/audit_records.jsonl with source SHA-256 metadata; no policy calls.
- Saved reachable IK gate passed. Exact Open3D query-thread equivalence is separately recorded; single-thread queries inside per-part worker processes avoid repeated small-query parallel setup. No geometry, precision, tolerance or sampling change.
- User-authorized update: B preparation runs separately in scratch/p004b-ui (detached, no new plan branch); final B verification/commits wait for A. Audit scopes are VAL first (primary interim), then train baseline/tail, with train final last/optional and explicitly unmeasured if omitted. Hash-verified complete geometry snapshots preserve prior calculations when rescheduling. Retract/ENV v5 remains out of scope.
- A mandatory scopes completed and combined in results/collision_audit_v1.md/.json; optional final-weight train is unmeasured. Three actual local Viser collision screenshots/PDFs/receipts are in results/visualization/. Source quality registry, existing figures and frozen demo remain unchanged; README/test report receive only the requested generated limitation sentence.
- DNS push failure from the preceding request is separate from this plan; no push or remote-history operation is authorized here.
- Future ENV v5: predeclare retract intermediate poses, dense transfer checking and recomputed transfer-time matrices; this requires regenerating every affected result. Not implemented now.

## 5. Closure

- Audit completed for 1095 VAL and 3400 primary train episodes. The optional 2400 final-weight train episodes were not measured and do not enter rates. Saved IK and executed scan sample gates passed; paths, geometry, quality results and closed holdout are unchanged.
- Offline BSD-3-Clause UR5 visuals passed both flange/TCP checks at 1870 saved poses: maximum position difference 0.300000000896 mm, axis difference 2.41483653945e-06 degrees; no fitted joint offsets. Capsules remain the distinct collision model.
- Recorded comparison uses two cells and a shared simulated clock. 21 immutable VAL cases passed exact trajectory/time checks; the completed browser inspection and four attributed screenshots are indexed by results/visualization/polish_screenshots_manifest_v1.json. Collision depths are sampled interval maxima; unsafe paths are displayed without correction.
- Six independent slide sets (PNG/PDF/JSON) and their manifest are in results/figures/slides/. Existing report figures and frozen demo bytes/hashes remain unchanged. The generated Interpretation in results/collision_audit_v1.md states the robot.movej_time fixed-endpoint/rest-to-rest lower bound and explicitly limits normalized-ratio and test extrapolations; it changes no hypothesis.
- D-071 and the ENV v5 proposal are versioned. Historical evidence is preserved; results/provenance/commit_map_v1.csv resolves referenced rewritten commits separately. Project documents received only neutral editorial substitutions with existing numerical literals preserved. Private workflow files stay excluded. The detached preparation worktree was removed after nonrecursive junction unlinking; root data/docs/outputs remain present.

### Open issues carried to Plan 003

- Source license versions for the three VAL demo parts remain unrecorded; do not infer a version or imply resolved distribution licensing. Existing attribution/change notices are retained.
- Retract poses, dense transition checking and recomputed time matrices are future ENV v5 work; they require a new protocol and regenerated results. No Phase 2 work is activated by this closure.
