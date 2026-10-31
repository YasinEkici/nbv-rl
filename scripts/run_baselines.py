"""Run all Plan 001 baselines on train and val variants; write outputs/<run_id>/ and registry rows.

    uv run python scripts/run_baselines.py                 # 48 and 128 candidates, train + val
    uv run python scripts/run_baselines.py --sizes 48 --methods greedy_cost ilp_tsp

Test parts are never run here. Stochastic methods use seeds 0-9; deterministic ones one run.
Rows are written with status `exploratory` until the protocol is frozen (Plan 001 Slice 6).
"""

import argparse
import csv
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nbv import baselines, evaluation, parts
from nbv.env import ENV_VERSION, NBVEnv, evaluate_policy, evaluate_sequence

STOCHASTIC = {"random", "random_tsp", "mcts"}
OPEN_LOOP = {"fixed", "ilp_tsp", "random_tsp"}
METHODS = ["random", "fixed", "greedy_ig", "greedy_cost", "greedy_oracle", "ilp_tsp", "random_tsp", "mcts"]
INFO = {"random": "none", "fixed": "nominal", "greedy_ig": "nominal", "greedy_cost": "nominal",
        "greedy_oracle": "real", "ilp_tsp": "nominal", "random_tsp": "nominal", "mcts": "nominal"}


def d_plan95(path: str, n: int) -> dict[int, int]:
    return {int(r["base"]): int(r[f"d_plan95_n{n}"]) for r in csv.DictReader(open(path, encoding="utf-8"))}


def run_task(task: dict) -> dict:
    method, n, split, seed, variants = task["method"], task["n"], task["split"], task["seed"], task["variants"]
    env = NBVEnv(variants, n_candidates=n)
    rng = np.random.default_rng([seed, n, METHODS.index(method)])
    repair = baselines.repair_policy(env.v, env.a)
    kplan = d_plan95(task["d_plan"], n)
    plans: dict[int, tuple[list[int], dict]] = {}
    episodes, t0 = [], time.perf_counter()
    for vid in variants:
        fid = int(vid.split("_")[0])
        info: dict = {}
        if method in OPEN_LOOP:
            env.reset(options={"variant": vid})
            if method == "random_tsp":  # random per variant and seed
                plan, info = baselines.random_tsp_plan(env.nominal, math.ceil(1.5 * kplan[fid]), rng)
            else:  # deterministic, depends only on the nominal CAD
                if fid not in plans:
                    plans[fid] = (baselines.fixed_pattern(env.nominal), {}) if method == "fixed" else baselines.ilp_tsp_plan(env.nominal)
                plan, info = plans[fid]
            res = evaluate_sequence(env, vid, plan, repair)
            info = {**info, "plan": plan, "success_without_repair": evaluate_sequence(env, vid, plan)["success"]}
        else:
            make = {
                "random": lambda e: baselines.random_policy(rng),
                "greedy_ig": lambda e: baselines.greedy_ig_action,
                "greedy_cost": lambda e: baselines.greedy_cost_policy(e.v, e.a),
                "greedy_oracle": lambda e: baselines.greedy_oracle_policy(e.data.real),
                "mcts": lambda e: baselines.MCTSPlanner(e.v, e.a, rng, e.k_max, e.target),
            }[method]
            res = evaluate_policy(env, vid, make)
        m = evaluation.episode_metrics(res)
        episodes.append({"run_id": task["run_id"], "method": method, "variant": vid, "part": fid, "split": split,
                         "seed": seed, "n_candidates": n, **m, "sequence": [s["pass"] for s in res["steps"]],
                         "repair": [s["repair"] for s in res["steps"]], "steps": res["steps"], "return": res["return"],
                         "censored": res["censored"], "planner": info})
    return {"task": task, "episodes": episodes, "wall_s": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--sizes", nargs="+", type=int, default=[48, 128])
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--d-plan", default="results/d_plan_v1.csv")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--registry", default="results/registry.csv")
    args = ap.parse_args()
    manifest = parts.load_yaml("configs/split_v1.yaml")
    tasks = []
    for method in args.methods:
        for n in args.sizes:
            for split in args.splits:
                variants = [v["variant_id"] for v in manifest["variants"] if v["split"] == split]
                for seed in (range(args.seeds) if method in STOCHASTIC else [0]):
                    run_id = f"p001s5-{method}-n{n}-{split}-s{seed}"
                    tasks.append({"method": method, "n": n, "split": split, "seed": seed, "variants": variants,
                                  "run_id": run_id, "d_plan": args.d_plan})
    rows, commit = [], evaluation.git_commit()
    with ProcessPoolExecutor(args.workers) as pool:
        for out in pool.map(run_task, tasks):
            t, eps = out["task"], out["episodes"]
            summary = evaluation.aggregate(eps)
            extra = {
                "info": INFO[t["method"]], "loop": "open+repair" if t["method"] in OPEN_LOOP else "closed",
                "n_candidates": t["n"], "parts_version": "v2", "wall_s": round(out["wall_s"], 2),
                **{k: round(summary[k], 4) for k in ("cycle_time_median_s", "repair_passes", "nsc_at_5", "nsc_at_10",
                                                     "raw_at_5", "raw_at_10", "total_joint_motion_rad")},
                "n_episodes": summary["n_episodes"],
            }
            if t["method"] in OPEN_LOOP:
                extra["success_without_repair"] = round(float(np.mean([e["planner"]["success_without_repair"] for e in eps])), 4)
            cfg = {k: t[k] for k in ("method", "n", "split", "seed")} | {"env": "configs/env_v1.yaml", "parts": "configs/parts_v2.yaml",
                                                                           "split_manifest": "configs/split_v1.yaml"}
            prov = {"commit": commit, "config_hash": evaluation.config_hash(cfg), "env_version": ENV_VERSION,
                    "split_version": "v1", "parts_version": "v2"}
            evaluation.write_run(Path(args.outputs) / t["run_id"], cfg, prov, eps, summary)
            seed = t["seed"] if t["method"] in STOCHASTIC else ""
            rows.append(evaluation.registry_row(t["run_id"], t["method"], f"n{t['n']}", t["split"], seed, summary, extra,
                                                ENV_VERSION, "v1"))
            print(f"{t['run_id']:40s} success {summary['success_rate']:.3f} cycle {summary['cycle_time_to_95_s']:.1f}s")
    evaluation.append_registry(Path(args.registry), rows)
    print(f"appended {len(rows)} rows to {args.registry}")


if __name__ == "__main__":
    main()
