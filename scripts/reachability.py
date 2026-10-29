"""Scan passes + IK + PTP times for every variant; cache plans and write the reachability report.

    uv run python scripts/reachability.py
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import trimesh

from nbv import candidates, parts, robot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default="configs/parts_v1.yaml")
    ap.add_argument("--manifest", default="configs/split_v1.yaml")
    ap.add_argument("--sensor", default="configs/sensor_v1.yaml")
    ap.add_argument("--candidates", default="configs/candidates_v1.yaml")
    ap.add_argument("--robot", default="configs/robot_v1.yaml")
    ap.add_argument("--cache", default="data/cache/robot")
    ap.add_argument("--out", default="results/reachability_v1.csv")
    args = ap.parse_args()

    pc, manifest = parts.load_yaml(args.parts), parts.load_yaml(args.manifest)
    sensor, cc, rc = (parts.load_yaml(p) for p in (args.sensor, args.candidates, args.robot))
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for var in manifest["variants"]:
        mesh = trimesh.load(Path(pc["paths"]["variants"]) / f"{var['variant_id']}.ply", process=False)
        for n in (cc["n_candidates"], cc["n_candidates_sensitivity"]):
            passes = candidates.make_passes(
                mesh.vertices, n, cc["min_elevation_deg"], candidates.standoff_offset(sensor), cc["sweep_margin_mm"]
            )
            plan = robot.plan_passes(passes, rc, sensor["scan_speed_mm_s"])
            np.savez_compressed(
                cache / f"{var['variant_id']}_n{n}.npz",
                center=passes.center, direction=passes.direction, view=passes.view, sweep=passes.sweep,
                standoff=passes.standoff, length=passes.length, reachable=plan.reachable, q=plan.q,
                ptp=plan.ptp, from_home=plan.from_home, to_home=plan.to_home, scan_time=plan.scan_time,
            )
            rows.append({
                "variant_id": var["variant_id"], "base": var["base"], "split": var["split"], "n_candidates": n,
                "n_reachable": int(plan.reachable.sum()), "reachable_ratio": round(float(plan.reachable.mean()), 4),
                "observable_reachable_over_all": "",  # filled in Slice 3 (visibility)
            })
        print(var["variant_id"], [r["n_reachable"] for r in rows[-2:]])
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
