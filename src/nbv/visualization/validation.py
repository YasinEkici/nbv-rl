"""Final presentation checks: saved train/VAL poses and immutable VAL trajectories only."""
import json
from pathlib import Path
import numpy as np
from nbv import evaluation,parts
from nbv.visualization.demo import check_bundle
from nbv.visualization.live import recorded_session
from nbv.visualization.playback import RecordedTimeline,SharedClock,AuditIndex
from nbv.visualization.replay import digest,require_exact
from nbv.visualization.ur5_visual import verify_saved_poses


def verify(directory='results/visualization'):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    model_path=directory/'urdf_alignment_v1.json';target=directory/'polish_validation_v1.json'
    if model_path.exists() or target.exists():raise FileExistsError('verification receipts already exist; use --output-dir scratch/<new-directory>')
    provenance=evaluation.git_state()|dict(code_sha256=evaluation.code_hash())
    if provenance['dirty']:raise ValueError('commit presentation code before final verification')
    registry=digest('results/registry.csv');bundle=check_bundle('results/demo')
    model=verify_saved_poses(model_path)
    audit_path=Path('results/collision_audit_v1.json');audit=AuditIndex(json.loads(audit_path.read_text(encoding='utf-8')))
    cases=[];clocks=[]
    for part in parts.load_yaml('configs/split_v1.yaml')['parts']:
        if part['split']!='val':continue
        fid=part['file_id'];loaded=recorded_session(fid,'greedy_cost');data=loaded['part_data'];tracks=[]
        for episode in data['episodes']:
            track=RecordedTimeline(episode['steps']);tracks.append(track)
            count=audit.validate_timeline(episode['source_run_id'],episode['variant'],track)
            require_exact(track.duration,episode['summary']['cycle_time'],'recorded final cycle time')
            cases.append(dict(part=fid,variant=episode['variant'],method=episode['method'],seed=episode['seed'],
                source_run=episode['source_run_id'],audited_colliding_segments_matched=count,
                original_oracle=episode['verification'],geometry_endpoints_and_time_bounds_exact=True))
        pair=SharedClock(tracks[:2]);times=sorted({0.,*[r['cumulative_time_s'] for e in data['episodes'][:2] for r in e['steps']]})
        for t in times:
            pair.seek(t);states=pair.states()
            require_exact(states[0]['simulation_time_s'],states[1]['simulation_time_s'],'comparison global time')
            if not all(np.isfinite(s['q']).all() for s in states):raise ValueError('nonfinite recorded animation pose')
        clocks.append(dict(part=fid,shared_time_checks=len(times),all_exact=True))
    if digest('results/registry.csv')!=registry or check_bundle('results/demo')!=bundle:raise ValueError('protected source changed')
    result=dict(schema_version=1,provenance=provenance,urdf_enabled=model['passed'],urdf_report_sha256=digest(model_path),
        audit_sha256=digest(audit_path),registry_sha256=registry,bundle=bundle,cases=cases,comparison=clocks,
        all_recorded_paths_exact=True,no_policy_calls=True,no_test_access=True,no_quality_registry_writes=True,
        fallback='If URDF alignment fails, retain capsules; never use an unmatched visual model.')
    target.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result
