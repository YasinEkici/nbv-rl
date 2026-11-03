"""Joint-optimum reference (D-035): one CP-SAT model choosing and ordering passes, per base part.

    uv run python scripts/joint_opt.py              # train + val parts, 48 candidates, 300 s each

Writes plans to data/cache/plans/joint_opt_n{n}.json (used by scripts/run_baselines.py) and the
planner-objective comparison with ILP + TSP to results/joint_opt_v2.csv. Test parts are not run.
"""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from nbv import baselines, parts
from nbv.env import NBVEnv


def solve(task: dict) -> dict:
    fid, n, tl, workers = task["fid"], task["n"], task["time_limit"], task["workers"]
    env = NBVEnv([f"{fid}_v0"], n_candidates=n)
    env.reset(options={"variant": f"{fid}_v0"})
    nom = env.nominal
    ilp, ilp_info = baselines.ilp_tsp_plan(nom)
    plan, info = baselines.joint_opt_plan(nom, time_limit_s=tl, hint=ilp, workers=workers)
    c_ilp, c_joint = baselines.planner_cost(nom, ilp), baselines.planner_cost(nom, plan)
    return {
        "base": fid, "split": task["split"], "n_candidates": n, "plan": plan, "ilp_plan": ilp,
        "ilp_passes": len(ilp), "joint_passes": len(plan), "ilp_tsp_cost_s": round(c_ilp, 3),
        "joint_cost_s": round(c_joint, 3), "joint_bound_s": round(info["joint_bound_s"], 3),
        "joint_status": info["joint_status"], "joint_wall_s": round(info["joint_wall_s"], 1),
        "ilp_status": ilp_info["ilp_status"], "tsp": ilp_info["tsp"],
        # Planner-objective gap of ILP + TSP to the joint optimum, bracketed by CP-SAT's bound.
        "gap_min_pct": round(100 * (c_ilp - c_joint) / c_ilp, 2),
        "gap_max_pct": round(100 * (c_ilp - info["joint_bound_s"]) / c_ilp, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/joint_opt_v2.csv")
    ap.add_argument("--plans", default="data/cache/plans")
    args = ap.parse_args()
    manifest = parts.load_yaml("configs/split_v1.yaml")
    tasks = [{"fid": p["file_id"], "split": p["split"], "n": args.n, "time_limit": args.time_limit, "workers": args.workers}
             for p in manifest["parts"] if p["split"] in ("train", "val")]
    with ProcessPoolExecutor(args.parallel) as pool:
        rows = list(pool.map(solve, tasks))
    for r in rows:
        print({k: v for k, v in r.items() if k not in ("plan", "ilp_plan")})
    Path(args.plans).mkdir(parents=True, exist_ok=True)
    plan_path = Path(args.plans) / f"joint_opt_n{args.n}.json"
    plan_path.write_text(json.dumps({str(r["base"]): r["plan"] for r in rows}, indent=1), encoding="utf-8")
    out = Path(args.out)
    fields = [k for k in rows[0] if k not in ("plan", "ilp_plan")]
    existing = [r for r in csv.DictReader(open(out, encoding="utf-8"))] if out.exists() else []
    existing = [r for r in existing if int(r["n_candidates"]) != args.n]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(existing + [{k: r[k] for k in fields} for r in rows])
    print(f"wrote {out} and {plan_path}")


if __name__ == "__main__":
    main()
