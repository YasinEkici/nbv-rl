"""Robot base position re-check with collision masks (D-027, D-042); train parts, 48 candidates.

    uv run python scripts/base_check.py

Predeclared rule: base x in {-300, -350, -400, -450} mm; criterion = train mean of
observable_reachable_over_all (reachable union / all-pass union of each train variant, cached
dual-occlusion visibility, which does not depend on the base); the best position replaces -350 only
if it is better by more than 0.01 (stability rule). Also reported per position: IK-reachable passes,
passes masked by the collision check, reachable passes.
"""

import argparse
import copy
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nbv import candidates, parts, robot, visibility

CURRENT = -350.0


def part_rows(task: dict) -> list[dict]:
    a, fid, x = task["args"], task["fid"], task["x"]
    pc, sensor, cc, rc = (parts.load_yaml(p) for p in (a["parts"], a["sensor"], a["candidates"], a["robot"]))
    rc = copy.deepcopy(rc)
    rc["base_xyz_mm"][0] = x
    g = pc["geometry"]
    mesh = parts.load_base(Path(pc["paths"]["raw"]) / f"{fid}.ply", g["longest_mm"], g["max_edge_mm"])
    n = cc["n_candidates"]
    passes = candidates.make_passes(mesh.vertices, n, cc["min_elevation_deg"], candidates.standoff_offset(sensor), cc["sweep_margin_mm"])
    plan = robot.plan_passes(passes, rc, sensor["scan_speed_mm_s"], part_mesh=mesh)
    rows = []
    for vid in task["variants"]:
        meas = visibility.Visibility.load(Path(a["vis_cache"]) / f"{vid}_n{n}.npz").measured()
        u_all, u_reach = visibility.union(meas), visibility.union(meas, plan.reachable)
        rows.append({"base_x_mm": x, "base": fid, "variant_id": vid, "n_candidates": n,
                     "n_ik_reachable": int(plan.ik_reachable.sum()),
                     "n_collision_masked": int((plan.ik_reachable & ~plan.reachable).sum()),
                     "n_reachable": int(plan.reachable.sum()),
                     "observable_reachable_over_all": round(float(u_reach.sum() / max(u_all.sum(), 1)), 4),
                     "observable_ik_reachable_over_all": round(float(visibility.union(meas, plan.ik_reachable).sum()
                                                                     / max(u_all.sum(), 1)), 4)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default="configs/parts_v2.yaml")
    ap.add_argument("--manifest", default="configs/split_v1.yaml")
    ap.add_argument("--sensor", default="configs/sensor_v1.yaml")
    ap.add_argument("--candidates", default="configs/candidates_v1.yaml")
    ap.add_argument("--robot", default="configs/robot_v2.yaml")
    ap.add_argument("--vis-cache", default="data/cache/visibility")
    ap.add_argument("--positions", type=float, nargs="+", default=[-300.0, -350.0, -400.0, -450.0])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="results/base_check_v2.csv")
    args = ap.parse_args()
    a = {k: getattr(args, k) for k in ("parts", "sensor", "candidates", "robot", "vis_cache")}
    manifest = parts.load_yaml(args.manifest)
    train = [p["file_id"] for p in manifest["parts"] if p["split"] == "train"]
    tasks = [{"x": x, "fid": fid, "args": a,
              "variants": [v["variant_id"] for v in manifest["variants"] if v["base"] == fid]}
             for x in args.positions for fid in train]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for rs in pool.map(part_rows, tasks) for r in rs]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}\n")
    print(f"{'base x':>7} {'IK reach.':>9} {'coll. masked':>12} {'reachable':>9} {'IK ratio':>8} {'reach. ratio':>12} "
          f"{'obs. ratio mean':>15} {'min':>6} {'obs. ratio, IK only':>19}")
    mean = {}
    for x in args.positions:
        rs = [r for r in rows if r["base_x_mm"] == x]
        per_part = {r["base"]: r for r in rs}.values()  # pass counts are per base part
        n = sum(r["n_candidates"] for r in per_part)
        ik, masked, reach = (sum(r[k] for r in per_part) for k in ("n_ik_reachable", "n_collision_masked", "n_reachable"))
        mean[x] = float(np.mean([r["observable_reachable_over_all"] for r in rs]))
        print(f"{x:7.0f} {ik:9d} {masked:12d} {reach:9d} {ik / n:8.3f} {reach / n:12.3f} {mean[x]:15.3f} "
              f"{min(r['observable_reachable_over_all'] for r in rs):6.3f} "
              f"{np.mean([r['observable_ik_reachable_over_all'] for r in rs]):19.3f}")
    best = max(mean, key=mean.get)
    choice = best if mean[best] - mean[CURRENT] > 0.01 else CURRENT
    print(f"\nbest {best:.0f} mm ({mean[best]:.4f}) vs current {CURRENT:.0f} mm ({mean[CURRENT]:.4f}): "
          f"difference {mean[best] - mean[CURRENT]:+.4f} -> rule chooses {choice:.0f} mm")


if __name__ == "__main__":
    main()
