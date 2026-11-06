"""Prepared only: do not run until the user separately authorizes opening the test split."""
import nbv
import argparse
from nbv.test_evaluation import open_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--i-understand-this-opens-test', action='store_true')
    args = parser.parse_args()
    open_test(understand=args.i_understand_this_opens_test)
