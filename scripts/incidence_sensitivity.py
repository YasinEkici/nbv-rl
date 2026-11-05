"""Incidence-angle sensitivity at the visibility level (D-012): 60, 65, 70 degrees; no method runs.

    uv run python scripts/incidence_sensitivity.py

Train and val variants, 48 and 128 candidates, nominal passes (D-029). For each angle the
dual-occlusion visibility is recomputed with the laser incidence limit at that angle ("laser"), and
with both the laser and the camera limit at that angle ("laser+camera", since D-026 ties the camera
limit to the laser's). Reported per variant: measurable surface share (reachable union / samples,
the coverage denominator of D-023), all-pass union / samples, and the relative laser loss of H3
(surface the camera-only model counts but the dual model does not, / dual union). Nothing is cached.
"""

import argparse
import copy
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nbv  # establish thread defaults before NumPy in every spawned worker
import numpy as np
import trimesh

from nbv import parts, visibility
from nbv.candidates import ScanPasses

ANGLES = (60.0, 65.0, 70.0)


def variant_rows(task: dict) -> list[dict]:
    a = task["args"]
    pc, sensor, vc = (parts.load_yaml(p) for p in (a["parts"], a["sensor"], a["visibility"]))
    vid, fid = task["variant"], task["base"]
    mesh = trimesh.load(Path(pc["paths"]["variants"]) / f"{vid}.ply", process=False)
    s = np.load(Path(a["vis_cache"]) / f"samples_{fid}.npz")
    p, nrm = visibility.SurfaceSamples(s["face"], s["bary"]).on(mesh)
    scene = visibility.make_scene(mesh, mesh.vertices[:, 2].min() - vc["table_offset_mm"])
    rows = []
    for n in a["sizes"]:
        z = np.load(Path(a["robot_cache"]) / f"{fid}_n{n}.npz")
        passes, reach = ScanPasses.from_cache(z), z["reachable"]
        for angle in ANGLES:
            for mode in ("laser", "laser+camera"):
                if angle == 65.0 and mode == "laser+camera":
                    continue  # identical to "laser" at the main threshold
                v = copy.deepcopy(vc)
                v["max_laser_incidence_deg"] = angle
                if mode == "laser+camera":
                    v["max_camera_angle_deg"] = angle
                g = visibility.SensorGeometry.from_config(sensor, v)
                rep = visibility.coverage_report(visibility.compute_visibility(scene, p, nrm, passes, g, v["ray_offset_mm"]), reach)
                rows.append({"variant_id": vid, "base": fid, "split": task["split"], "n_candidates": n, "angle_deg": angle,
                             "mode": "main" if angle == 65.0 else mode,
                             **{k: round(float(rep[k]), 4) for k in ("reachable_union_over_samples", "union_all_over_samples",
                                                                      "laser_occlusion_loss_rel")}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default="configs/parts_v2.yaml")
    ap.add_argument("--manifest", default="configs/split_v1.yaml")
    ap.add_argument("--sensor", default="configs/sensor_v1.yaml")
    ap.add_argument("--visibility", default="configs/visibility_v1.yaml")
    ap.add_argument("--robot-cache", default="data/cache/robot/nominal")
    ap.add_argument("--vis-cache", default="data/cache/visibility")
    ap.add_argument("--sizes", type=int, nargs="+", default=[48, 128])
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--out", default="results/incidence_sensitivity_v2.csv")
    args = ap.parse_args()
    a = {k: getattr(args, k) for k in ("parts", "sensor", "visibility", "robot_cache", "vis_cache", "sizes")}
    manifest = parts.load_yaml(args.manifest)
    tasks = [{"variant": v["variant_id"], "base": v["base"], "split": v["split"], "args": a}
             for v in manifest["variants"] if v["split"] in ("train", "val")]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for rs in pool.map(variant_rows, tasks) for r in rs]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} rows ({len(tasks)} train + val variants)")
    print(f"\n{'n':>4} {'mode':13} {'angle':>5}  {'reach. union/samples':>21} {'all union/samples':>18} {'rel. laser loss (H3)':>21}")
    for n in args.sizes:
        for mode, angle in (("laser", 60.0), ("main", 65.0), ("laser", 70.0), ("laser+camera", 60.0), ("laser+camera", 70.0)):
            rs = [r for r in rows if r["n_candidates"] == n and r["mode"] == mode and r["angle_deg"] == angle]
            m = {k: np.mean([r[k] for r in rs]) for k in ("reachable_union_over_samples", "union_all_over_samples", "laser_occlusion_loss_rel")}
            print(f"{n:4d} {mode:13} {angle:5.0f}  {m['reachable_union_over_samples']:21.3f} {m['union_all_over_samples']:18.3f} "
                  f"{m['laser_occlusion_loss_rel']:21.3f}")


if __name__ == "__main__":
    main()
