"""Check the manuscript against registered/stored sources, without experiments."""
import argparse
import json
from pathlib import Path
import nbv
from nbv.manuscript import render,verify

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true',help='Update only README.md from the public manuscript template')
    parser.add_argument('--claims',help='Write the numerical source/formula audit to a new file')
    args=parser.parse_args()
    if args.write:Path('README.md').write_text(render()[0],encoding='utf-8')
    if args.claims:
        target=Path(args.claims);target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('x',encoding='utf-8') as stream:json.dump(render()[1],stream,ensure_ascii=False,indent=2)
    print(json.dumps(verify(),ensure_ascii=False,indent=2))
