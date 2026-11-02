"""How often the hierarchical bootstrap interval contains the true mean, on synthetic nested data (D-039).

    uv run python scripts/bootstrap_coverage.py                  # 3, 8, 11 parts, 200 datasets each

Data: y = 5 + part effect (sd 1) + variant effect (sd 0.3) + seed noise (sd 0.1); 5 variants per part,
3 seeds per variant, as in the part split. Nominal level 0.95. With few parts the percentile
interval of a bootstrap over parts is too narrow, so its coverage stays below the nominal level.
First run (2026-11-02, sanity check): 0.72 (3 parts), 0.89 (8), 0.86 (11).
"""

import argparse
import json

import numpy as np

from nbv.evaluation import hierarchical_bootstrap


def coverage(n_parts: int, reps: int, n_boot: int, rng: np.random.Generator) -> dict:
    hit, widths = 0, []
    for rep in range(reps):
        d = {}
        for p in range(n_parts):
            pe = rng.normal(0, 1.0)
            d[f"p{p}"] = {f"v{v}": list(5.0 + pe + rng.normal(0, 0.3) + rng.normal(0, 0.1, size=3)) for v in range(5)}
        r = hierarchical_bootstrap(d, n_boot=n_boot, seed=rep)
        hit += r["ci_low"] <= 5.0 <= r["ci_high"]
        widths.append(r["ci_high"] - r["ci_low"])
    return {"coverage_of_true_mean": hit / reps, "median_width": round(float(np.median(widths)), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", type=int, nargs="+", default=[3, 8, 11])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = {n: coverage(n, args.reps, args.n_boot, rng) for n in args.parts}
    print(json.dumps({"nominal": 0.95, "by_n_parts": out}, indent=1))


if __name__ == "__main__":
    main()
