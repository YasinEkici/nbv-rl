"""Planning difficulty per base part: fewest passes covering 100% / 95% of the nominal reachable union.

    uv run python scripts/d_plan.py

Uses only the nominal CAD, so it is computed for every part including test parts; test values are
never used for any tuning decision (PROTOCOL).
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from nbv import baselines, parts, visibility


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="configs/split_v1.yaml")
    ap.add_argument("--robot-cache", default="data/cache/robot/nominal")
    ap.add_argument("--vis-cache", default="data/cache/visibility/nominal")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--out", default="results/d_plan_v2.csv")
    args = ap.parse_args()
    manifest = parts.load_yaml(args.manifest)
    rows = []
    for p in manifest["parts"]:
        fid = p["file_id"]
        row = {"base": fid, "split": p["split"], "stratum": p["stratum"], "hard_ratio": p["hard_ratio"]}
        for n in (48, 128):
            reach = np.load(Path(args.robot_cache) / f"{fid}_n{n}.npz")["reachable"]
            vis = visibility.Visibility.load(Path(args.vis_cache) / f"{fid}_n{n}.npz").measured()
            ref, r = visibility.union(vis, reach), np.flatnonzero(reach)
            for frac, tag in ((1.0, "d_plan"), (0.95, "d_plan95")):
                res = baselines.set_cover(vis, ref, r, frac, args.time_limit)
                row[f"{tag}_n{n}"] = len(res.passes)
                row[f"{tag}_n{n}_status"] = res.status
                row[f"{tag}_n{n}_gap"] = round(res.gap, 4)
        rows.append(row)
        print(fid, p["split"], {k: v for k, v in row.items() if k.startswith("d_plan") and "_status" not in k and "_gap" not in k})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
