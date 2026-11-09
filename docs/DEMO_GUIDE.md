# Playing the NBV demos

Presentation outputs: [slide gallery](../results/figures/slides/README.md), [recorded comparison screenshot](../results/visualization/polish_comparison_v1.png) and [capture provenance](../results/visualization/polish_screenshots_manifest_v1.json). Historical source revisions are resolved separately by the [commit map](../results/provenance/commit_map_v1.csv); the frozen demo bundle is unchanged.

Run from the repository root in PowerShell. The viewer is local and uses **VAL only**. It offers two modes: play the already exported data, or run the frozen policies live. Neither mode trains a model or writes experiment results. Test data stays closed.

## 1. Start with the recorded demo

This is the simplest presentation path. It needs the committed bundle and the optional viewer packages; original meshes, visibility caches and checkpoints are not needed. The application still uses the repository's fixed sensor config for the schematic laser fan.

```powershell
Set-Location -LiteralPath 'D:\Python Projects\nbv-rl'
uv sync --extra demo
uv run --extra demo python scripts/live_demo.py --recorded --port 8080
```

Open the URL printed in the terminal, normally **http://127.0.0.1:8080**. Keep the terminal running. Stop the server with **Ctrl+C** when finished. If the requested port is occupied, use the actual port printed by Viser or choose another with `--port`.

The recorded player checks every file against `results/demo/manifest.json` before loading the scene. It refuses a hash/size mismatch. It does not re-export or modify the bundle. The exact upload list and serialized sizes are in [the generated handoff](../results/visualization/README.md); [bundle format](../results/demo/README.md) describes the arrays.

Choose a part and method. Recorded mode contains only variant **v0**, seed **0**. Click **Başlat / Duraklat** to play a whole episode, including its return home. Use **Adım** to advance one scan pass; after the last scan, one additional step plays **eve dönüş**. Click **Sıfırla** to return to the initial scene. Selecting a new part/method also resets playback.

## 2. Run the live demo

Use the same installation, without `--recorded`:

```powershell
uv run --extra demo python scripts/live_demo.py --port 8080
```

Live mode needs the original allowed VAL caches and canonical `outputs/` artifacts already present on the experiment machine. A source-only clone should use recorded mode. Do not regenerate missing experiment inputs merely to open a demo.

| Selector | Available choices |
|---|---|
| Parça | Configured VAL parts 271875, 591247, 37275 |
| Varyant | v0–v4 |
| Yöntem | greedy_cost, mcts, sarsa/linear, sarsa/value_estimate, regq/linear, ilp_tsp, joint_opt |
| Seed | 0–9 for MCTS/learners; 0 for deterministic baselines |

Greedy/MCTS and the three learners make live decisions with the frozen settings. Learners use the official tail-average weights and deterministic argmax (`epsilon = 0`); checkpoint identity and file hashes are checked against the pretest manifest before inference. `value_estimate` names how that policy was trained; the potential is not recomputed during frozen inference.

ILP+TSP and joint optimum are labeled **Kayıtlı plan**: their stored VAL sequences are executed without solving again. Opening a later MCTS variant can take preparation time because the original RNG stream spans the ordered VAL variants. Earlier canonical VAL episodes are reproduced to restore that stream; preparation states may be cached in memory.

## 3. Controls and scene

| Control | Meaning |
|---|---|
| Başlat / Duraklat | Start automatic playback or pause after the current pass |
| Adım | Animate one pass, or the separate return-home step |
| Sıfırla | Cancel playback, clear covered samples and return to q_home |
| Hız (×) | Change animation speed; recorded times/metrics do not change |
| 48 aday pası göster | Toggle saved candidates, colored by reachability/IK/collision reason |

Drag the scene to orbit; use the mouse wheel to zoom. Scene controls are shared by connected local clients. Use one operator during a presentation.

Both modes use the same point colors: **blue** = previously measured; **orange** = newly measured in this pass; **light blue** = reachable but not yet measured; **gray** = unmeasurable. Coverage changes only after a complete pass, following ENV v4; the moving laser fan is schematic. The normalized nominal mesh and selected-variant sample points intentionally differ slightly. Their sample correspondence is preserved.

The UR5 link capsules and sensor box use the recorded collision geometry and ENV v4 joint states. Motion between saved states is interpolated for display only. It is not a checked robot trajectory. The renderer sends float32 display coordinates to the browser; the immutable bundle and numerical metrics retain their original precision.

The status panel shows step/pass, accumulated estimated cycle time, primary coverage and verification. Primary coverage is relative to the reachable surface union, not all surface samples. Thus high primary coverage can coexist with many gray/unmeasurable samples. The return home adds time but no coverage; do not stop the presentation before that final segment.

## 4. Read verification and timing correctly

Before completion the live panel says there is no matching claim yet. At the end it compares the produced sequence and full-precision summary with the original VAL episode. **Kayıtla birebir ✓** means exact matching; **UYUŞMAZLIK** explicitly lists differing fields. Do not hide the warning or edit source records/hashes to make it pass. Report it and use the already validated recorded mode for a separate demonstration.

Recorded mode reports the package SHA-256 check and preserves the original per-episode verification. It does not claim a fresh independent policy evaluation. Learner final coverage/joints are derived and checked against the available partial oracle; they were not independently stored as historical step traces.

Live timing is labeled **canlı, bu makine, görüntüleme yükü dahil**. It includes step/decision and scene-update work, excluding deliberate animation waits. It is not comparable to the D-066 benchmark. Recorded/CAD playback has no new decision-time measurement.

## 5. A short presentation walkthrough

1. Open recorded mode, keep the default VAL part, and select greedy_cost. Start with the candidate overlay hidden so the part and robot remain clear.
2. Advance one pass. Point out the movement, scan, newly measured samples and primary-coverage label.
3. Play to completion, including return home. Show the hash/verification message and the attribution below it.
4. Reset and select SARSA/linear, then SARSA/value_estimate. These are the predetermined policies, not choices made after viewing a test result. Use the registered figures for quantitative comparisons.
5. If demonstrating live inference, start live mode in place of recorded mode, select an allowed seed and wait for the exact-match result. Distinguish live decisions from the two recorded CAD plans.

Static figures are available immediately, without running a server: open [coverage versus time](../results/figures/p004_d_val_coverage_time.png), [the corrected candidate map](../results/figures/p004_g_val_candidate_sphere.png), or [the complete figure index](../results/visualization/README.md).

## 6. Troubleshooting

| Symptom | Action |
|---|---|
| `No module named viser` | Use `uv sync --extra demo` and **`uv run --extra demo`**; plain uv run may remove the optional extra |
| Missing original checkpoint/cache in live mode | Use `--recorded`; inspect the missing VAL source separately |
| Hash/identity mismatch | Stop that load; restore the exact original artifact through the normal review process, not edited manifests |
| Blank/loading scene | Read the GUI error/status and terminal; check the local URL and browser WebGL support |
| Slow MCTS initialization | Allow its original RNG-prefix preparation to finish; choose recorded mode for uninterrupted presentation |
| Target coverage reached but no final check | Play the final **eve dönüş** row |
| Exporter says directory exists | Exporting is not playback; do not rerun `export_demo.py` to start the viewer |

Keep the attribution panel visible when sharing a mesh/derivative. It includes title, author, source, recorded CC family, missing-version status and changes. License versions for these parts are **not recorded**; no specific deed version has been inferred. Resolve that source metadata before public redistribution. No public sharing/tunnel is started by this viewer.

## 7. Polished viewer and side-by-side comparison

Use the same `uv sync --extra demo` and `live_demo.py` commands above. The demo extra includes the pinned COLLADA reader and chart package; base installation is unchanged. UR5 geometry is vendored in `assets/ur5/` with BSD-3-Clause license/source receipts, so no model download is needed during a presentation. Plotly JavaScript is served locally from the installed package, and the viewer disables external environment-map assets.

In **Oynat**, select the part and method, then use **Başlat / Duraklat**, **Adım**, **Sıfırla**, speed and the simulated-time slider. **Genel / Parça / Yan** are camera presets. The coverage bar marks the target. **48 aday pası göster** toggles candidates; **Çarpışma modeli** toggles the capsules. These are distinct from the UR5 visual meshes. At an audited collision the implicated collision body is red even if the other capsules are hidden; the warning does not repair or hide the path.

The warning depth is the greatest sampled penetration within the active recorded collision interval, not a new per-frame geometric computation. Interval boundaries come from the dense audit and do not constitute a continuous collision-free guarantee.

In recorded mode, open **Karşılaştır**, enable **İki hücreyi karşılaştır**, and choose the second method. Return to **Oynat** to run both cells with one simulated clock. The shorter episode waits at home after finishing. The chart highlights the first method and shows the other recorded methods faintly. The cell offset is a display transform only. Recorded data stays at v0/seed 0; the live mode still offers configured VAL variants/seeds, and does not run a second policy for comparison.

**Bilgi** contains point-color meanings, nominal/variant coordinate notes and full part/robot attribution. Compact attribution is also visible in the play panel. Learner coverage retains its partial-oracle provenance label. URDF loading requires the committed saved-pose check; a missing, mismatched or failed check produces an explicit capsule fallback. The source URDF's base-frame conversion is retained without fitting joint offsets or changing the frozen robot model.

Audit: [recorded-path collision report](../results/collision_audit_v1.md). Presentation-only [figures and receipts](../results/figures/slides/README.md) do not replace the report figures. Neither playback mode opens the test holdout. Do not rerun exporters or edit the demo manifest to start a viewer.

Theme v2 supplies current method and semantic colors. The immutable demo common.json styles are historical v1; playback does not modify the bundle. Palette measurement dependencies are not required to run the viewer.
