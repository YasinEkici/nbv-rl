"""Scan passes + IK + PTP times per variant; cache plans and write the reachability report.

    uv run python scripts/reachability.py
    uv run python scripts/reachability.py --base-x -450 --split train --sizes 48 \\
        --cache data/cache/robot/base_x-450 --out results/base_sweep/base_x-450.csv

Unreachable passes are counted by region: `far` = direction pointing away from the robot
(u_x > 0), `robot_side` = towards the robot (u_x <= 0).
"""

import argparse
import copy
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
    ap.add_argument("--base-x", type=float, help="override robot base x (mm); diagnostic sweeps only")
    ap.add_argument("--split", choices=["train", "val", "test"], help="only variants of this split")
    ap.add_argument("--sizes", type=int, nargs="+", help="candidate-set sizes (default: main + sensitivity)")
    ap.add_argument("--cache", default="data/cache/robot")
    ap.add_argument("--out", default="results/reachability_v1.csv")
    args = ap.parse_args()

    pc, manifest = parts.load_yaml(args.parts), parts.load_yaml(args.manifest)
    sensor, cc, rc = (parts.load_yaml(p) for p in (args.sensor, args.candidates, args.robot))
    if args.base_x is not None:
        rc = copy.deepcopy(rc)
        rc["base_xyz_mm"][0] = args.base_x
    sizes = args.sizes or [cc["n_candidates"], cc["n_candidates_sensitivity"]]
    variants = [v for v in manifest["variants"] if args.split in (None, v["split"])]
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for var in variants:
        mesh = trimesh.load(Path(pc["paths"]["variants"]) / f"{var['variant_id']}.ply", process=False)
        for n in sizes:
            passes = candidates.make_passes(
                mesh.vertices, n, cc["min_elevation_deg"], candidates.standoff_offset(sensor), cc["sweep_margin_mm"]
            )
            plan = robot.plan_passes(passes, rc, sensor["scan_speed_mm_s"])
            np.savez_compressed(
                cache / f"{var['variant_id']}_n{n}.npz",
                center=passes.center, direction=passes.direction, view=passes.view, sweep=passes.sweep,
                standoff=passes.standoff, length=passes.length, reachable=plan.reachable, q=plan.q,
                ptp=plan.ptp, from_home=plan.from_home, to_home=plan.to_home, scan_time=plan.scan_time,
                ptp_pi=plan.ptp_pi, from_home_pi=plan.from_home_pi, to_home_pi=plan.to_home_pi,
            )
            far = passes.direction[:, 0] > 0
            dq = np.degrees(plan.leading_dq())
            dq_pi = np.degrees(plan.leading_dq(wrapped=True))
            rows.append({
                "variant_id": var["variant_id"], "base": var["base"], "split": var["split"],
                "base_x_mm": rc["base_xyz_mm"][0], "n_candidates": n,
                "n_reachable": int(plan.reachable.sum()), "reachable_ratio": round(float(plan.reachable.mean()), 4),
                "unreachable_far": int((~plan.reachable & far).sum()),
                "unreachable_robot_side": int((~plan.reachable & ~far).sum()),
                "dq_lead_min_deg": round(float(dq.min()), 1), "dq_lead_median_deg": round(float(np.median(dq)), 1),
                "dq_lead_max_deg": round(float(dq.max()), 1), "n_ptp_over_270deg": int((dq > 270).sum()),
                "dq_lead_pi_median_deg": round(float(np.median(dq_pi)), 1), "dq_lead_pi_max_deg": round(float(dq_pi.max()), 1),
                "observable_reachable_over_all": "",  # filled in Slice 3 (visibility)
            })
        print(var["variant_id"], [r["n_reachable"] for r in rows[-len(sizes):]])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
