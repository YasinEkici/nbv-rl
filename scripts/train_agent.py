"""Train Phase 1 agents on train and evaluate VAL; no test-part interface."""

import nbv  # numerical thread defaults before worker imports
import argparse

from nbv.experiment import METHODS, grid_settings, run_tasks, smoke, tuning_tasks
from nbv.shaping import PHIS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--method", choices=METHODS, default="sarsa")
    ap.add_argument("--features", type=int, choices=(11, 16), default=11)
    ap.add_argument("--phi", choices=PHIS, default="linear")
    ap.add_argument("--alpha", type=float, default=.03)
    ap.add_argument("--lam", type=float, default=.9)
    ap.add_argument("--eta", type=float, default=.001)
    ap.add_argument("--phi-alpha", type=float, default=50.)
    ap.add_argument("--epsilon", choices=("linear", "constant"), default="linear")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--budget", type=int, default=2000000)
    ap.add_argument("--status", choices=("exploratory", "canonical"), default="exploratory")
    ap.add_argument("--tag", default="p002")
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    if args.smoke:
        smoke()
    elif args.grid:
        run_tasks(tuning_tasks(), status="exploratory", tag="p002t", workers=args.workers)
    else:
        setting = {"alpha": args.alpha, "lam": args.lam, "eta": args.eta,
                   "phi_alpha": args.phi_alpha, "epsilon": args.epsilon}
        tasks = [{"method": args.method, "features": args.features, "phi": args.phi,
                  "setting": setting, "seed": seed, "budget": args.budget} for seed in args.seeds]
        run_tasks(tasks, status=args.status, tag=args.tag, workers=args.workers)


if __name__ == "__main__":
    main()
