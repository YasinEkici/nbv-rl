# Plan 004 S0: read-only visualization data inventory

Date: 2026-11-08. Source snapshot before this plan: main 01090f9. Active documentation branch at this inventory: plan/004-visualization. Related: D-063/D-064/D-066/D-068 and [completed Plan 004](../plans/done/004-visualization.md). Inventory statements below retain their S0 context; D-069/D-070 record subsequent implementation.

This is inventory, not an exported dataset or a new evaluation. Reads were limited to repository code/config/docs, registry, existing VAL outputs, the three selected VAL nominal/variant-0 caches and meshes, installed API metadata and public source/package metadata. **No real test cache, test episode output or environment was opened. Inventory commands constructed no environment of any split and performed no replay, policy choice, ray tracing, IK, decimation, demo export or package install.** The separately required existing pytest suite still uses its normal synthetic/train/VAL fixtures; this is not a visualization/evaluation run. Small Python snippets inspected existing files in memory; no application/test implementation was added.

## 1. S1/S2 feasibility matrix

| Item | Already recorded | To derive later, only after approval | Status |
|---|---|---|---|
| S1a quality–latency | Test episode metrics in registry extra_json; D-066 p002planning timings in registry | Join the five policy labels and calculate summaries from records; log-ms axis; VAL/seed-0/hardware caption | Available; drawing pending |
| S1b run-wise curves | Every canonical p002 row has 20 history values; original training.json repeats them; ten seeds per arm | Rank by final recorded value; highlight complete actual near-5/50/95-percentile runs and show all seeds, with deterministic tie handling | Available; drawing pending |
| S1c per-part points | Test registry episode_metrics has variant, part, method, seed, weight kind and normalized time | Panels/dots from primary tail rows plus baseline rows; preserve actual seed counts | Available; registry only |
| S1d coverage–time | VAL baseline step arrays have coverage/times; VAL learners have saved sequences and aggregate checks | Baselines can use saved steps; learner curves require allowed sequence reconstruction and a clarified independent final-coverage check | Baselines available; learner publication gate |
| S1e time components | Test registry has cycle_time and total_joint_motion | Cannot identify move/scan/home components from those totals | Requested decomposition impossible under current registry-only rule; total time/joint motion remain available |
| S1f gallery | Allowed raw/nominal meshes, sample points/normals, nominal visibility, reachable masks | Count measured reachable passes per sample; classify table-facing/never-measured; render fixed legends and attribution | Derivable from VAL/train caches; no ray tracing needed |
| S1g pass sphere | Robot cache has direction, view, sweep, standoff, length, reachable and ik_reachable | Start/end positions and three reason categories; coordinate-frame annotations | Derivable; no new IK needed |
| S2 mesh | All three raw nominal meshes are below 50000 triangles | Normalize with the same source transform; optional display-only decimation of a subdivided representation | Available; export pending |
| S2 samples/visibility | Nominal points/normals, face/barycentric correspondence, variant-0 mesh and five packed visibility layers | Compose measured bits; optionally derive variant sample coordinates from shared correspondence; label both coordinate meanings | Available/derivable; coordinate choice explicit before export |
| S2 pass/robot data | Pass geometry, q[48,5,6], masks, robot/TCP/home/collision configs | DH vectors from installed UR5 model, JSON typing/units; executed lifted joints require exact v4 transfer logic | Derivable; historical executed joints lack independent episode references |
| S2 episodes | All 21 requested source sequences exist: seven methods × three VAL variant-0 parts | Baseline steps already present; learners need reconstruction and validation gates, no fresh policy evaluation by default | Source sequences available; full learner validation unresolved |
| S2 schema/manifest/size | Source identities and existing manifests, no demo bundle yet | Versioned schema, base64/bit conventions, per-episode checks, SHA-256, visible attribution, actual serialized byte count | Future implementation/test work; not done in S0 |
| S3 Viser | Package metadata and successful binary-only dry resolution | Approved optional group, actual installation and browser/server startup | Candidate only; not installed/runtime-tested |

## 2. Episode records: learners versus baselines

### Learner VAL records

Canonical p002 VAL tail episodes are in outputs/<source_run>/episodes.jsonl. Each source file has all 15 VAL variants. The checked seed-0 learner source files are approximately 9.6–9.7 kB each. Fields:

```text
variant, part, method, seed, split, run_id, sequence,
cycle_time, normalized_cycle_time, success, censored,
passes_executed, passes_to_95, repair_passes, total_joint_motion,
coverage_at_5, coverage_at_10, nsc_at_5, nsc_at_10,
raw_at_5, raw_at_10, auc_coverage_time
```

There are **no steps, per-step move/scan/home times, per-step coverage, q states or separate final_coverage**. coverage_at_10 is final only when the episode ends by that step; the selected sequences below are longer, so it cannot be renamed final coverage. Original training.json contains a secondary final-weight episode list with sequences/aggregates, also without step traces. A sequence is not missing merely because steps are missing: recorded-pass replay is sufficient in principle, so invoking a learner for new actions is not justified by this inventory.

### Baseline VAL records

Plan 001 canonical VAL files have the same episode metrics plus sequence, steps, repair, planner and return. Four checked method files contain 15 variants each, approximately 74–78 kB. Each saved step contains:

```text
pass, t_move, t_scan, t_home, joint_motion, total_joint_motion,
cycle_time, coverage, coverage_nsc, coverage_raw,
nominal_coverage, repair
```

Thus baseline intermediate and final coverage/time references exist. Executed q states were not saved. Robot q_plan is a separate nominal five-sample plan; actual lifted joints must follow the original wrap/limit logic, not be asserted equal to an absent historical q field. Return-home time is included in a terminal step's t_home/cycle_time; S2 must split out an explicit home row without double counting.

### Exact seed-0 source identities

| Policy | Registered source run_id | Sequence lengths for 271875_v0 / 591247_v0 / 37275_v0 | Stored steps |
|---|---|---|---|
| SARSA/linear | p002-sarsa-val-s0-v4-20261106T142107-41f52b15 | 14 / 12 / 12 | No |
| SARSA/value_estimate | p002-sarsa-val-s0-v4-20261106T150038-99429658 | 13 / 12 / 12 | No |
| RegQ/linear | p002-regq-val-s0-v4-20261106T142115-fff73794 | 13 / 12 / 12 | No |
| greedy_cost | p001s6-greedy_cost-n48-val-s0-v4-20261103T170747-69917d | 13 / 13 / 12 | Yes |
| MCTS | p001s6-mcts-n48-val-s0-v4-20261103T170747-790316 | 12 / 11 / 12 | Yes |
| ILP+TSP | p001s6-ilp_tsp-n48-val-s0-v4-20261103T170747-177728 | 12 / 12 / 12 | Yes |
| joint_opt | p001s6-joint_opt-n48-val-s0-v4-20261103T170747-705797 | 12 / 12 / 12 | Yes |

Deterministic baseline registry seed cells are empty; their s0 run/config identity is available. Stochastic MCTS uses registry seed 0. Never interpret empty cells as missing data requiring a new run.

## 3. Learning histories and test registry limits

- Canonical p002: 60 VAL rows, six arms × ten official seeds. Every extra_json.history has the 20-point 100000…2000000 step grid, normalized_cycle_time, success_rate, weight_norm and other summaries. It preserves each run individually.
- outputs/<run>/training.json history repeats those values. It also stores policy_weights, final_weights, episode_seeds, secondary and tail_count. Original files lack later-added identity fields; bind by their config/registry/manifests as D-057 specifies.
- results/val_curves_v1.json contains seven arm-level series (six canonical and one exploratory full-feature), each with 20 steps/mean/low/high. It **does not retain the ten individual trajectories**. S1b must use registry history or original training.json, not try to recover seeds from mean/min/max. The exploratory arm remains outside test comparisons.
- All 145 test registry rows use a compact episode_metrics key set: part, variant, method, seed, success, censored, cycle_time, normalized_cycle_time, passes_to_95, coverage_at_5/10, auc_coverage_time, total_joint_motion, repair_passes, nsc_at_5/10, raw_at_5/10, success_without_repair. Weight kind/Phi/source_run live in the surrounding extra_json for learner rows.
- Test registry has **no sequence, steps, t_move, t_scan, t_home or component totals**, for either learners or baselines. Existing raw baseline test output might contain more, but Plan 004 does not read it. The requested test decomposition is not identifiable under the rule; no estimate or total-time subtraction will be presented as measured components.
- D-066 decision timings are already registered in the single exploratory p002planning row. Use that row and its definition, not a new benchmark. A test-quality versus VAL-latency plot needs an explicit cross-split caption.

## 4. VAL variant-0 caches and meshes

All paths below are the three explicitly allowed VAL ids, never a test id. Files were read in memory without exporting derived arrays. Dimensions/bytes are file/array metadata, not new quality results.

| Part | Variant visibility NPZ bytes | Nominal visibility NPZ bytes | Nominal points NPZ bytes | Sample-correspondence NPZ bytes | Robot NPZ bytes |
|---|---:|---:|---:|---:|---:|
| 271875 | 344118 | 344161 | 746625 | 516145 | 25436 |
| 591247 | 271103 | 270152 | 624480 | 519212 | 25428 |
| 37275 | 264069 | 264817 | 460051 | 516585 | 25460 |

Path patterns:

```text
data/cache/visibility/{id}_v0_n48.npz
data/cache/visibility/nominal/{id}_n48.npz
data/cache/visibility/nominal/points_{id}.npz
data/cache/visibility/samples_{id}.npz
data/cache/robot/nominal/{id}_n48.npz
data/raw/parts/{id}.ply
data/cache/variants_v2/{id}_v0.ply
```

The visibility NPZs hold n_samples=20000 and geom/laser_angle/laser_los arrays [48,2500] uint8, plus cam_angle/cam_los [1,48,2500] uint8. Each layer is 120000 uncompressed packed bytes. nbv.visibility.Visibility.load unpacks them; measured() AND-composes laser/camera conditions into [48,20000] booleans. A packed measured-only matrix is 120000 bytes, base64 160000 bytes. It does not require new ray tracing.

Nominal points and normals are each [20000,3] float64: 480000 bytes apiece, 960000 combined, base64 1280000 combined. Correspondence is face[20000] int64 plus bary[20000,3] float64. Variant positions/normals can be derived from the stored variant mesh via this correspondence; nominal points must not silently stand in for variant geometry.

Robot q is [48,5,6] float64, 11520 bytes (15360 base64). Other fields: center, direction/view/sweep, standoff, length, reachable, ik_reachable, min_clearance, scan_time, ptp_pi, from_home_pi and to_home_pi. All three inspected parts have 34 reachable, 9 IK-unreachable and 5 collision-masked passes; all reachable q entries are finite. Unreachable entries require validity masks. Cached geometry plus ScanPasses.samples(5) determines positions along a pass; no inverse-kinematics solve is needed.

DH is not an array in these NPZs. nbv.robot builds the installed Robotics Toolbox DH UR5 model; S2 can extract its six-link a/d/alpha/offset values with library/version provenance. Base/TCP/home come from configs/robot_v2.yaml; capsule/sensor-box geometry comes from configs/collision_v1.yaml. Full executed joint states still require the original transfer/wrap logic.

| Part | Raw nominal vertices | Raw nominal triangles | Raw PLY bytes | Variant-0 vertices | Variant-0 triangles | Variant-0 PLY bytes |
|---|---:|---:|---:|---:|---:|---:|
| 271875 | 10976 | 22012 | 418087 | 192350 | 384760 | 7310301 |
| 591247 | 1812 | 3696 | 70009 | 420053 | 840178 | 15963171 |
| 37275 | 9293 | 18594 | 353456 | 318024 | 636056 | 12085237 |

The display can use the raw nominal mesh with the same normalization transform; simulation subdivision need not be exported. All three raw nominal meshes already satisfy the approximate 50000-triangle target. Open3D 0.20.0 is installed and exposes TriangleMesh.simplify_quadric_decimation(target_number_of_triangles,...), so optional display-only simplification is available without adding a dependency. No decimation or display export was run; runtime/visual quality of simplification is not established by this API inspection.

## 5. Attribution and distribution

| Part | Name | Author | License | Source |
|---|---|---|---|---|
| 271875 | PHIL | Mr_MegaTronic | CC BY | https://www.thingiverse.com/thing:136235 |
| 591247 | 3D printed mechanical Clock with Anchor Escapement | TheGoofy | CC BY-SA | https://www.thingiverse.com/thing:328569 |
| 37275 | R. Maker - Mark I | Mr_MegaTronic | CC BY-SA | https://www.thingiverse.com/thing:11654 |

These fields come from configs/split_v1.yaml. Every displayed model/bundle part needs its own visible attribution/license, including BY-SA derivative terms. A repository-wide software license does not replace those terms.

## 6. Viser suitability: check only, no install

Interpreter inspected: CPython 3.14.4, Windows AMD64. Viser was absent before and remains uninstalled. PyPI metadata lists Python >=3.8 and a Python 3.14 classifier; current source permits trimesh <6, compatible with the installed major version. Sources: https://pypi.org/project/viser/ and https://github.com/viser-project/viser/blob/main/pyproject.toml.

The following was run as an explicit non-installing check:

```text
uv pip install --dry-run --no-cache --python .venv/Scripts/python.exe --only-binary :all: viser

Resolved 20 packages in 682ms
Would download 8 packages
Would install 8 packages
 + imageio==2.38.1
 + markdown-it-py==4.2.0
 + mdurl==0.1.2
 + msgspec==0.22.0
 + rich==14.3.4
 + viser==1.1.1
 + websockets==16.1.1
 + zstandard==0.25.0
```

The first sandboxed attempt could not connect to PyPI (Windows socket error 10013); the same dry-run check with network access succeeded. No package, optional group or lockfile was changed. This demonstrates dependency/wheel resolution in the current environment, **not** tested installation, server/client startup or browser behavior. Those remain unverified until optional S3 is approved. Rerun/marimo are not adopted or installed.

## 7. Size and effort estimates

All sizes below are decimal bytes/MB, not MiB. They are estimates from inspected arrays, not a serialized demo. For the existing raw nominal meshes, float64 positions plus uint32 triangle indices require respectively 703424, 117120 and 594880 base64 bytes, total 1415424. One position/normal sample set per part adds 3840000 base64 bytes over three parts; three measured matrices add 480000; q arrays add 46080. Pass geometry, DH/configs, 21 step sequences, units, JSON keys and attribution are small additional fields.

- One labeled sample-coordinate set per part: approximately 6–7 MB including JSON/metadata allowance.
- Both nominal and deformed-variant position/normal sets: approximately 10–11 MB including allowance, still plausibly below 12 MB for these original meshes.
- These are not guarantees. S2 must measure the whole results/demo/ bundle plus manifest and reject >12000000 bytes. Encoding all five visibility layers, expanded boolean JSON or the huge variant meshes can break the cap. Preserve full-precision physics; do not silently quantize to fit.

Engineering estimates: S1 4–6 hours; S2 5–8 hours; combined 9–14 hours after approval, excluding gate-resolution/review and the external viewer. These are planning estimates, not registry experiment timings. Viewer work is scheduled 9–10 November; optional S3 is not included. S1e and independent learner final-coverage validation require the user's next decision before their affected outputs can be published.

## 8. S0 evidence and next step

- Plan-number check: only template, draft 003 and archived 001/002 existed; 004 was free. Live final-analysis references move to 005 (D-068); archived historical context remains intact.
- Only documentation is changed by S0. Existing result rows, selected settings, weights, frozen protocol/configs, claims and outputs remain unchanged. Full pytest runs validate the unchanged codebase; document checks first failed while the plan/inventory files were absent, then pass after creation.
- Stop here. The user must approve S1–S2 and decide the two missing-data gates; no renderer/exporter or replay has been started.
