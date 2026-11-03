"""Robot-part and robot-table clearance of every reachable pass (diagnostic, D-042; masks unchanged).

    uv run python scripts/collision_diagnostic.py
    uv run python scripts/collision_diagnostic.py --figure-part 46262 --figure scratch/collision/46262.png

Nominal mesh per base part, train and val parts only, all 5 IK samples of each reachable pass.
Transfers between passes are not checked. Writes one row per pass to results/collision_v1.csv and
prints a summary per part.
"""

import argparse
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nbv import collision, parts, robot

FIELDS = ["base", "split", "n_candidates", "pass", "u_x", "u_y", "u_z", "elevation_deg", "side", "height",
          "part_clearance_mm", "part_body", "table_clearance_mm", "table_body"]


def part_rows(task: dict) -> list[dict]:
    fid, split, n, args = task["fid"], task["split"], task["n"], task["args"]
    pc, rc, cc = (parts.load_yaml(p) for p in (args["parts"], args["robot"], args["collision"]))
    g = pc["geometry"]
    mesh = parts.load_base(Path(pc["paths"]["raw"]) / f"{fid}.ply", g["longest_mm"], g["max_edge_mm"])
    part = collision.PartDistance(mesh)
    geom = collision.ArmGeometry.from_config(cc)
    box = geom.box_points()
    arm = robot.make_robot(rc)
    z = np.load(Path(args["robot_cache"]) / f"{fid}_n{n}.npz")
    rows = []
    for i in np.flatnonzero(z["reachable"]):
        best_p, best_t = (np.inf, ""), (np.inf, "")
        for q in z["q"][i]:
            for body, (cp, ct) in collision.pose_clearance(arm, q, part, geom, box).items():
                best_p = min(best_p, (cp, body))
                best_t = min(best_t, (ct, body))
        u = z["direction"][i]
        el = float(np.degrees(np.arcsin(np.clip(u[2], -1, 1))))
        rows.append({
            "base": fid, "split": split, "n_candidates": n, "pass": int(i),
            "u_x": round(float(u[0]), 4), "u_y": round(float(u[1]), 4), "u_z": round(float(u[2]), 4),
            "elevation_deg": round(el, 1), "side": "robot_side" if u[0] <= 0 else "far_side",
            "height": "low" if el < 45 else "high",
            "part_clearance_mm": round(best_p[0], 2), "part_body": best_p[1],
            "table_clearance_mm": round(best_t[0], 2), "table_body": best_t[1],
        })
    return rows


def summary(rows: list[dict], near: float) -> None:
    def clr(r):
        return min(float(r["part_clearance_mm"]), float(r["table_clearance_mm"]))

    for n in sorted({r["n_candidates"] for r in rows}):
        rs = [r for r in rows if r["n_candidates"] == n]
        c = np.array([clr(r) for r in rs])
        print(f"\n== {n} candidates: {len(rs)} reachable passes (train + val)")
        q = [0, .05, .25, .5, .75, 1]
        print("quantiles 0/5/25/50/75/100% (mm): min clearance", np.round(np.quantile(c, q), 1))
        print("   to the part ", np.round(np.quantile([float(r["part_clearance_mm"]) for r in rs], q), 1))
        print("   to the table", np.round(np.quantile([float(r["table_clearance_mm"]) for r in rs], q), 1))
        print(f"{'part':>7} {'split':5} {'reach':>5} {'<0':>4} {'<0 %':>6} {'<10':>4} {'<10 %':>6}  bodies (<0)")
        for fid in dict.fromkeys(r["base"] for r in rs):
            pr = [r for r in rs if r["base"] == fid]
            cc = np.array([clr(r) for r in pr])
            bodies = Counter((r["part_body"] if float(r["part_clearance_mm"]) <= float(r["table_clearance_mm"])
                              else r["table_body"] + "(table)") for r, x in zip(pr, cc) if x < 0)
            print(f"{fid:>7} {pr[0]['split']:5} {len(pr):5d} {(cc < 0).sum():4d} {100 * (cc < 0).mean():5.1f}% "
                  f"{(cc < near).sum():4d} {100 * (cc < near).mean():5.1f}%  {dict(bodies)}")
        bad = [r for r in rs if clr(r) < 0]
        print("colliding passes by side / height:", dict(Counter((r["side"], r["height"]) for r in bad)))
        print("all reachable passes by side / height:", dict(Counter((r["side"], r["height"]) for r in rs)))
        print("colliding with the part / the table:",
              sum(float(r["part_clearance_mm"]) < 0 for r in rs), "/", sum(float(r["table_clearance_mm"]) < 0 for r in rs))


def figure(fid: int, n: int, rows: list[dict], args: dict, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pc, rc, cc = (parts.load_yaml(p) for p in (args["parts"], args["robot"], args["collision"]))
    g = pc["geometry"]
    mesh = parts.load_base(Path(pc["paths"]["raw"]) / f"{fid}.ply", g["longest_mm"], g["max_edge_mm"])
    geom = collision.ArmGeometry.from_config(cc)
    arm = robot.make_robot(rc)
    z = np.load(Path(args["robot_cache"]) / f"{fid}_n{n}.npz")
    rs = [r for r in rows if r["base"] == fid and r["n_candidates"] == n]
    clr = {r["pass"]: min(float(r["part_clearance_mm"]), float(r["table_clearance_mm"])) for r in rs}
    worst = min(clr, key=clr.get)
    pts = mesh.sample(4000, seed=0) if hasattr(mesh, "sample") else mesh.vertices[::50]
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(*pts.T, s=0.3, c="0.6")
    c0 = z["center"]
    for i, c in clr.items():
        col = "tab:red" if c < 0 else ("tab:orange" if c < cc["near_mm"] else "tab:green")
        ax.plot(*np.c_[c0, c0 + 260 * z["direction"][i]], color=col, lw=1.5 if c < cc["near_mm"] else 0.6)
    ax.set_title(f"{fid}, {n} candidates: red < 0 mm, orange < {cc['near_mm']:.0f} mm, green ok\n(lines: view directions of reachable passes)")
    ax.view_init(elev=25, azim=-60)
    ax.set_box_aspect((1, 1, 0.8))
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(*pts.T, s=0.3, c="0.6")
    q = z["q"][worst][len(z["q"][worst]) // 2]
    for name, r, a, b in collision.capsule_axes(arm, q, geom):
        ax2.plot(*np.c_[a, b], lw=max(r / 6, 1), alpha=0.5, solid_capstyle="round")
    T = arm.fkine(q).A.copy()
    T[:3, 3] *= 1000
    box = geom.box_points()[::7] @ T[:3, :3].T + T[:3, 3]
    ax2.scatter(*box.T, s=0.5, c="tab:red")
    ax2.set_title(f"worst pass {worst}: clearance {clr[worst]:.0f} mm (mid sweep sample)")
    lim = np.array([[-450, 250], [-300, 300], [0, 650]])
    ax2.set_xlim(*lim[0]), ax2.set_ylim(*lim[1]), ax2.set_zlim(*lim[2])
    ax2.set_box_aspect(lim[:, 1] - lim[:, 0])
    ax2.view_init(elev=20, azim=-110)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default="configs/parts_v2.yaml")
    ap.add_argument("--manifest", default="configs/split_v1.yaml")
    ap.add_argument("--robot", default="configs/robot_v2.yaml")
    ap.add_argument("--collision", default="configs/collision_v1.yaml")
    ap.add_argument("--robot-cache", default="data/cache/robot/nominal")
    ap.add_argument("--sizes", type=int, nargs="+", default=[48, 128])
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--out", default="results/collision_v1.csv")
    ap.add_argument("--figure-part", type=int, help="train part to draw (default: most colliding train part at 48)")
    ap.add_argument("--figure", default="scratch/collision/collisions.png")
    args = ap.parse_args()
    a = {k: getattr(args, k) for k in ("parts", "robot", "collision", "robot_cache")}
    manifest = parts.load_yaml(args.manifest)
    tasks = [{"fid": p["file_id"], "split": p["split"], "n": n, "args": a}
             for n in args.sizes for p in manifest["parts"] if p["split"] in ("train", "val")]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for rs in pool.map(part_rows, tasks) for r in rs]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")
    summary(rows, parts.load_yaml(args.collision)["near_mm"])
    n0 = min(args.sizes)
    fid = args.figure_part or max(
        (p["file_id"] for p in manifest["parts"] if p["split"] == "train"),
        key=lambda f: sum(min(r["part_clearance_mm"], r["table_clearance_mm"]) < 0 for r in rows
                          if r["base"] == f and r["n_candidates"] == n0))
    figure(fid, n0, rows, a, args.figure)


if __name__ == "__main__":
    main()
