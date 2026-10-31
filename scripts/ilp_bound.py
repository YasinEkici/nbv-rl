"""Exact max-coverage upper bound (real visibility, k passes) vs the coverage of deterministic baselines.

    uv run python scripts/ilp_bound.py

For each k, the CP-SAT max-coverage bound over the real variant's reachable passes is an upper
bound of every method's primary coverage after k passes (D-033). Train and val variants only.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from nbv import baselines
from nbv.env import NBVEnv, evaluate_policy, evaluate_sequence

VARIANTS = ["46262_v0", "124336_v0", "591247_v0", "37275_v0"]  # train, train, val, val


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--k-max", type=int, default=14)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--out", default="results/ilp_bound_v1.csv")
    args = ap.parse_args()
    rows = []
    for vid in args.variants:
        env = NBVEnv([vid], n_candidates=args.n)
        env.reset(options={"variant": vid})
        real, rows_r = env.data.real, np.flatnonzero(env.nominal.reachable)
        repair = baselines.repair_policy(env.v, env.a)
        seqs = {
            "greedy_ig": evaluate_policy(env, vid, lambda e: baselines.greedy_ig_action),
            "greedy_cost": evaluate_policy(env, vid, lambda e: baselines.greedy_cost_policy(e.v, e.a)),
            "greedy_oracle": evaluate_policy(env, vid, lambda e: baselines.greedy_oracle_policy(e.data.real)),
            "ilp_tsp": evaluate_sequence(env, vid, baselines.ilp_tsp_plan(env.nominal)[0], repair),
            "fixed": evaluate_sequence(env, vid, baselines.fixed_pattern(env.nominal), repair),
        }
        for k in range(1, args.k_max + 1):
            ub = baselines.max_coverage(real.visibility, real.union, rows_r, k, time_limit_s=30)
            row = {"variant": vid, "k": k, "ilp_bound": round(ub.value, 4), "ilp_status": ub.status}
            for name, res in seqs.items():
                st = res["steps"]
                cov = st[min(k, len(st)) - 1]["coverage"]
                assert cov <= ub.value + 1e-9, (vid, k, name)  # compare before rounding
                row[name] = round(cov, 4)
            rows.append(row)
            print(row)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
