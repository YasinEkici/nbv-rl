"""Run only the authorized Plan 002 train/VAL arms; never opens test parts."""
import nbv
import argparse
from nbv.official import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=20)
    run(parser.parse_args().workers)
