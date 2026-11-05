"""Plan 002 D-046 feature check; --benchmark measures full feature matrices on train parts."""

import nbv  # establish numerical thread defaults before loading multiprocessing/numerical code
import argparse

from nbv.feature_check import benchmark, run_check


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--repeats", type=int, default=10000)
    args = ap.parse_args()
    if args.benchmark:
        benchmark(args.repeats)
    else:
        run_check(workers=args.workers)


if __name__ == "__main__":
    main()
