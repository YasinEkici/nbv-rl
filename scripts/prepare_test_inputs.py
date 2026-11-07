"""Explicit incident-1 nominal-only preparation; never runs test episodes."""
import nbv  # numerical thread defaults before worker numerical imports
import argparse
import json
from pathlib import Path
from nbv.test_inputs import prepare_missing, dry_run, make_input_manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--point-output',default='scratch/incident1-point-determinism')
    parser.add_argument('--manifest',default='results/test_input_manifest_v1.json')
    args=parser.parse_args()
    result=prepare_missing(point_output=args.point_output,manifest_path=args.manifest)
    if not result.get('already_ready'):
        result['nominal_plans']={p.stem:json.loads(p.read_text(encoding='utf-8'))
                                 for p in Path('data/cache/plans').glob('test_*_n48.json')}
        Path('results/test_input_preparation_v1.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        report=dry_run()
        Path('results/test_input_inventory_after_v1.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        Path(args.manifest).write_text(json.dumps(make_input_manifest(report),indent=2),encoding='utf-8')
        print(f"Prepared {len(result['generated'])} inputs; missing: {len(report['missing'])}",flush=True)
