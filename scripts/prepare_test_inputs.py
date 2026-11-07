"""Explicit incident-1 nominal-only preparation; never runs test episodes."""
import nbv  # numerical thread defaults before worker numerical imports
import json
from pathlib import Path
from nbv.test_inputs import prepare_missing, dry_run, make_input_manifest

if __name__=='__main__':
    result=prepare_missing()
    Path('results/test_input_preparation_v1.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    report=dry_run()
    Path('results/test_input_inventory_after_v1.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    Path('results/test_input_manifest_v1.json').write_text(json.dumps(make_input_manifest(report),indent=2),encoding='utf-8')
    print(f"Prepared {len(result['generated'])} inputs; missing: {len(report['missing'])}",flush=True)
