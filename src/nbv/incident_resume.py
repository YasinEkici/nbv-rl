"""Explicitly authorized one-use continuation of the pre-environment incident (D-062)."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from nbv import evaluation
from nbv.test_inputs import dry_run, sha256, verify_input_manifest


def test_episode_files(outputs):
    """Detect output contamination without opening any held-out cache or episode content."""
    import yaml
    found=[]
    for episode in Path(outputs).glob('*/episodes.jsonl'):
        if 'test' in episode.parent.name.lower():
            found.append(str(episode)); continue
        config=episode.parent/'config.yaml'
        if config.is_file():
            data=yaml.safe_load(config.read_text(encoding='utf-8')) or {}
            if data.get('split',data.get('part_split'))=='test': found.append(str(episode))
    return found


def passing_suite():
    """Require a fresh complete green suite on the exact clean continuation snapshot."""
    run=subprocess.run(['uv','run','pytest','-q'],capture_output=True,text=True)
    return {'returncode':run.returncode,'output':run.stdout+run.stderr}


def resume_after_incident(*,incident,registry='results/registry.csv',outputs='outputs',
                          pretest_path='results/pretest_manifest_v1.json',
                          input_path='results/test_input_manifest_v1.json'):
    from nbv import test_evaluation as gate
    if incident!=1: raise ValueError('only incident 1 is authorized')
    outputs=Path(outputs)
    resumed=outputs/'p002-test-resumed-1.json'
    if resumed.exists(): raise FileExistsError('incident-1 continuation already claimed')
    opening=outputs/'p002-test-opened.json'
    record=outputs/'p002-test-incident-1.json'
    old=json.loads(opening.read_text(encoding='utf-8'))
    incident_record=json.loads(record.read_text(encoding='utf-8'))
    if (incident_record.get('incident')!=1 or incident_record.get('opening_commit')!='4850144'
            or old.get('commit')!='4850144' or sha256(opening)!=incident_record['opening_sha256']
            or incident_record.get('performance_information_produced') is not False
            or incident_record.get('evidence')!=dict(registry_test_rows=0,completion_marker_exists=False,test_episode_files=[])):
        raise ValueError('incident or original opening claim does not match the authorized event')
    if (any(r.get('part_split')=='test' for r in gate.load_rows(registry))
            or (outputs/'p002-test-completed.json').exists() or test_episode_files(outputs)):
        raise ValueError('test evaluation evidence already exists; cannot continue')
    selected,state=gate.preflight(understand=True,registry=registry,outputs=outputs,manifest_path=pretest_path)
    if state['dirty']: raise ValueError('continuation requires a clean tree')
    if ({r['run_id'] for r in selected}!=set(old['official_runs'])
            or sha256(pretest_path)!=old['pretest_manifest_sha256']):
        raise ValueError('original official run set or pretest manifest changed')
    for path in (pretest_path,input_path):
        evaluation._git('ls-files','--error-unmatch',str(path))
    inventory=dry_run(outputs=outputs)
    manifest=json.loads(Path(input_path).read_text(encoding='utf-8'))
    verify_input_manifest(manifest,inventory)
    # Ensure the incident and immutable original claim themselves are pinned in the committed input manifest.
    pinned={Path(e['path']).resolve():e['sha256'] for e in manifest['inputs']}
    if any(pinned.get(p.resolve())!=sha256(p) for p in (opening,record)):
        raise ValueError('incident/original claim are not bound by the input manifest')
    suite=passing_suite()
    if suite['returncode']!=0: raise ValueError('pytest failed before continuation:\n'+suite['output'])
    # Recheck source and all hashes immediately before the exclusive claim.
    fresh=evaluation.git_state()
    if fresh['dirty'] or fresh['commit']!=state['commit']: raise ValueError('source changed during continuation checks')
    verify_input_manifest(manifest,dry_run(outputs=outputs))
    state=state|dict(incident=1,original_opening_sha256=sha256(opening),incident_sha256=sha256(record),
        test_input_manifest_sha256=sha256(input_path),pytest=suite,
        resumed_at_utc=datetime.now(timezone.utc).isoformat(),prepared_plan_dir='data/cache/plans')
    gate.claim_opening(resumed,state)
    print('Incident-1 continuation claimed; all input, identity, clean-tree and pytest guards passed.',flush=True)
    return gate.execute(selected,state,registry,str(outputs))
