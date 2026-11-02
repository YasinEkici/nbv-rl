"""Run all Plan 001 baselines on train and val variants; write outputs/<run_id>/ and registry rows.

    uv run python scripts/run_baselines.py                          # 48 and 128 candidates, train + val
    uv run python scripts/run_baselines.py --status canonical --tag s6 --supersede

Test parts are never run here. Stochastic methods use seeds 0-9; deterministic ones one run.
Canonical runs need a clean working tree; run_ids are unique and outputs are never overwritten (D-037).
`joint_opt` needs data/cache/plans/joint_opt_n{n}.json from scripts/joint_opt.py (skipped otherwise).
Every episode gets normalized_cycle_time = cycle time / ILP + TSP cycle time on the same variant (D-034).
"""

import argparse
import csv
import json
import math
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nbv import baselines, evaluation, parts
from nbv.env import ENV_VERSION, NBVEnv, evaluate_policy, evaluate_sequence

STOCHASTIC = {"random", "random_tsp", "mcts"}
OPEN_LOOP = {"fixed", "ilp_tsp", "random_tsp", "joint_opt"}
METHODS = ["random", "fixed", "greedy_ig", "greedy_cost", "greedy_oracle", "ilp_tsp", "random_tsp", "mcts", "joint_opt"]
INFO = {"random": "none", "fixed": "nominal", "greedy_ig": "nominal", "greedy_cost": "nominal",
        "greedy_oracle": "real", "ilp_tsp": "nominal", "random_tsp": "nominal", "mcts": "nominal",
        "joint_opt": "nominal (solver-based upper reference, not an oracle)"}
ENV_CONFIG = "configs/env_v2.yaml"
# Every config the run or its precomputed caches depend on; hashed into the registry `config` column.
CONFIG_FILES = [ENV_CONFIG, "configs/robot_v1.yaml", "configs/parts_v2.yaml", "configs/split_v1.yaml",
                "configs/candidates_v1.yaml", "configs/sensor_v1.yaml", "configs/visibility_v1.yaml"]


def d_plan95(path: str, n: int) -> dict[int, int]:
    return {int(r["base"]): int(r[f"d_plan95_n{n}"]) for r in csv.DictReader(open(path, encoding="utf-8"))}


def run_task(task: dict) -> dict:
    method, n, split, seed, variants = task["method"], task["n"], task["split"], task["seed"], task["variants"]
    env = NBVEnv(variants, n_candidates=n, env_config=ENV_CONFIG)
    rng = np.random.default_rng([seed, n, METHODS.index(method)])
    repair = baselines.repair_policy(env.v, env.a)
    kplan = d_plan95(task["d_plan"], n)
    joint = json.loads(Path(task["joint_plans"]).read_text()) if method == "joint_opt" else {}
    plans: dict[int, tuple[list[int], dict]] = {}
    episodes, t0 = [], time.perf_counter()
    for vid in variants:
        fid = int(vid.split("_")[0])
        info: dict = {}
        if method in OPEN_LOOP:
            env.reset(options={"variant": vid})
            if method == "random_tsp":  # random per variant and seed
                plan, info = baselines.random_tsp_plan(env.nominal, math.ceil(1.5 * kplan[fid]), rng)
            elif method == "joint_opt":
                plan = joint[str(fid)]
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
    ap.add_argument("--status", default="exploratory", choices=["exploratory", "canonical"])
    ap.add_argument("--tag", default="s5", help="run-id tag, e.g. s5 (exploratory) or s6 (canonical)")
    ap.add_argument("--supersede", action="store_true", help="mark all earlier non-superseded plan 001 rows superseded")
    ap.add_argument("--d-plan", default="results/d_plan_v1.csv")
    ap.add_argument("--plans", default="data/cache/plans")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--registry", default="results/registry.csv")
    args = ap.parse_args()
    if "joint_opt" in args.methods and "ilp_tsp" not in args.methods:
        raise SystemExit("normalization needs ilp_tsp in --methods")
    state = evaluation.git_state()
    evaluation.check_status(args.status, state)
    files = evaluation.file_hashes(CONFIG_FILES)
    inputs = evaluation.file_hashes([args.d_plan, *map(str, sorted(Path(args.plans).glob("joint_opt_n*.json")))])
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    manifest = parts.load_yaml("configs/split_v1.yaml")
    tasks = []
    for method in args.methods:
        for n in args.sizes:
            joint_plans = Path(args.plans) / f"joint_opt_n{n}.json"
            if method == "joint_opt" and not joint_plans.exists():
                print(f"skip joint_opt n{n}: {joint_plans} missing")
                continue
            for split in args.splits:
                variants = [v["variant_id"] for v in manifest["variants"] if v["split"] == split]
                for seed in (range(args.seeds) if method in STOCHASTIC else [0]):
                    run_id = evaluation.new_run_id(f"p001{args.tag}-{method}-n{n}-{split}-s{seed}-{ENV_VERSION}", stamp)
                    tasks.append({"method": method, "n": n, "split": split, "seed": seed, "variants": variants,
                                  "run_id": run_id, "d_plan": args.d_plan, "joint_plans": str(joint_plans)})
    with ProcessPoolExecutor(args.workers) as pool:
        outs = list(pool.map(run_task, tasks))

    # Normalized cycle time against ILP + TSP on the same variant and candidate set (D-034).
    ref = {(e["variant"], e["n_candidates"]): e["cycle_time"] for o in outs if o["task"]["method"] == "ilp_tsp" for e in o["episodes"]}
    for o in outs:
        for e in o["episodes"]:
            e["normalized_cycle_time"] = e["cycle_time"] / ref[(e["variant"], e["n_candidates"])]

    if args.supersede:
        n_sup = evaluation.mark_superseded(Path(args.registry), lambda r: r["plan"] == "001" and r["status"] != "superseded")
        print(f"marked {n_sup} earlier rows superseded")
    rows = []
    for out in outs:
        t, eps = out["task"], out["episodes"]
        summary = evaluation.aggregate(eps)
        extra = {
            "info": INFO[t["method"]], "loop": "open+repair" if t["method"] in OPEN_LOOP else "closed",
            "n_candidates": t["n"], "parts_version": "v2", "wall_s": round(out["wall_s"], 2),
            **{k: round(summary[k], 4) for k in ("cycle_time_median_s", "normalized_cycle_time", "repair_passes", "nsc_at_5",
                                                 "nsc_at_10", "raw_at_5", "raw_at_10", "total_joint_motion_rad")},
            "n_episodes": summary["n_episodes"],
        }
        if t["method"] in OPEN_LOOP:
            extra["success_without_repair"] = round(float(np.mean([e["planner"]["success_without_repair"] for e in eps])), 4)
        params = {k: t[k] for k in ("method", "n", "split", "seed")}
        cfg = params | {"config_files": files}
        prov = {**state, "config_hash": evaluation.config_hash(cfg), "inputs": inputs, "env_version": ENV_VERSION,
                "split_version": "v1", "parts_version": "v2"}
        evaluation.write_run(Path(args.outputs) / t["run_id"], cfg, prov, eps, summary)
        seed = t["seed"] if t["method"] in STOCHASTIC else ""
        rows.append(evaluation.registry_row(t["run_id"], t["method"], evaluation.config_ref(files, {"n": t["n"]}), t["split"],
                                            seed, summary, extra, ENV_VERSION, "v1", evaluation.commit_label(state),
                                            status=args.status))
        print(f"{t['run_id']:44s} success {summary['success_rate']:.3f} cycle {summary['cycle_time_to_95_s']:.1f}s "
              f"norm {summary['normalized_cycle_time']:.3f}")
    evaluation.append_registry(Path(args.registry), rows)
    print(f"appended {len(rows)} rows to {args.registry}")


if __name__ == "__main__":
    main()
