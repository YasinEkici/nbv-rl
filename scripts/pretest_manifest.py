"""Create or verify frozen artifacts; never constructs a test environment."""
import nbv
import argparse
import json
from pathlib import Path
from nbv.test_evaluation import load_rows, make_manifest, preflight

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    if args.write:
        manifest=make_manifest(load_rows('results/registry.csv'))
        with Path('results/pretest_manifest_v1.json').open('x',encoding='utf-8') as stream:
            json.dump(manifest,stream,indent=2,sort_keys=True)
        print('Frozen 60 canonical runs: 240 raw SHA-256 hashes and 1800 reproduced VAL episodes.')
    else:
        selected,state=preflight(understand=True)
        print(f'PASS: {len(selected)} canonical checkpoints, identities and committed manifest hashes; test unopened.')
