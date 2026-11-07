"""Incident-1 input inventory and nominal-only preparation; no episodes or policies."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from nbv import parts, visibility
from nbv.env import NominalModel, _nominal_points
from nbv.experiment import eligible_cache_paths
from nbv.feature_check import CONFIG_FILES


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def required_inputs(*, variants=None, run_ids=None, outputs='outputs'):
    """Read only public split/checkpoint metadata to enumerate explicit input paths."""
    if variants is None:
        variants = [v['variant_id'] for v in parts.load_yaml('configs/split_v1.yaml')['variants'] if v['split']=='test']
    if run_ids is None:
        run_ids = list(json.loads(Path('results/pretest_manifest_v1.json').read_text(encoding='utf-8'))['runs'])
    entries=[]
    def add(path, role, part=None):
        entries.append(dict(path=Path(path).as_posix(), role=role, part=part))
    for path in eligible_cache_paths([], variants):
        role='nominal_points' if Path(path).name.startswith('points_') else 'visibility_or_robot'
        add(path, role)
    cfg=parts.load_yaml('configs/parts_v2.yaml')
    for fid in sorted({v.split('_')[0] for v in variants}):
        add(f'data/cache/visibility/samples_{fid}.npz','nominal_sample_correspondence',fid)
        add(Path(cfg['paths']['raw'])/f'{fid}.ply','nominal_CAD',fid)
        add(f'data/cache/plans/test_{fid}_n48.json','ilp_tsp_and_joint_plans_with_solver_info',fid)
    for path in [*CONFIG_FILES, 'results/d_plan_v2.csv', 'results/selected_settings_v1.json',
                 'results/pretest_manifest_v1.json', 'results/registry.csv']:
        add(path,'configuration_or_reference')
    for run in run_ids:
        for name in ('weights.npz','training.json','config.yaml','episodes.jsonl','provenance.json'):
            add(Path(outputs)/run/name,'canonical_checkpoint')
    add(Path(outputs)/'p002-test-opened.json','original_opening_claim')
    add(Path(outputs)/'p002-test-incident-1.json','incident_record')
    return entries


def dry_run(**kwargs):
    """No cache content reads, environment, solver, episode, evaluation or registry writes."""
    entries=[]
    for e in required_inputs(**kwargs):
        path=Path(e['path'])
        exists=path.is_file()
        entries.append(e|{'exists':exists,'size_bytes':path.stat().st_size if exists else None})
    return {'inputs':entries,'missing':[e['path'] for e in entries if not e['exists']]}


def print_inventory(report):
    for e in report['inputs']:
        print(f"{'PRESENT' if e['exists'] else 'MISSING'} {e['path']} ({e['size_bytes']} bytes; {e['role']})")
    print(f"Inputs: {len(report['inputs'])}; missing: {len(report['missing'])}",flush=True)


def array_content_hash(path):
    """Canonical array content, independent of ZIP headers or compression."""
    h=hashlib.sha256()
    with np.load(path) as data:
        for key in sorted(data.files):
            a=np.ascontiguousarray(data[key])
            h.update(f'{key}|{a.dtype}|{a.shape}'.encode())
            h.update(a.tobytes())
    return h.hexdigest()


def prove_point_determinism(destination):
    """Regenerate one TRAIN point cache using the same on-demand implementation."""
    import shutil
    split=parts.load_yaml('configs/split_v1.yaml')
    fid=next(p['file_id'] for p in split['parts'] if p['split']=='train')
    destination=Path(destination)
    (destination/'nominal').mkdir(parents=True,exist_ok=False)
    shutil.copyfile(f'data/cache/visibility/samples_{fid}.npz',destination/f'samples_{fid}.npz')
    _nominal_points(fid,destination,parts.load_yaml('configs/parts_v2.yaml'))
    original=f'data/cache/visibility/nominal/points_{fid}.npz'
    regenerated=destination/'nominal'/f'points_{fid}.npz'
    old,new=array_content_hash(original),array_content_hash(regenerated)
    if old!=new: raise ValueError('nominal point content determinism failed')
    return dict(train_part=fid,original=original,regenerated=str(regenerated),
                original_content_sha256=old,regenerated_content_sha256=new,exact_content_match=True)


def load_nominal(fid):
    """Load nominal-only arrays directly: never construct NBVEnv or read a real variant."""
    points,normals=_nominal_points(fid,Path('data/cache/visibility'),parts.load_yaml('configs/parts_v2.yaml'))
    measured=visibility.Visibility.load(Path(f'data/cache/visibility/nominal/{fid}_n48.npz')).measured()
    with np.load(f'data/cache/robot/nominal/{fid}_n48.npz') as z:
        reachable=z['reachable']
        return NominalModel(points=points,normals=normals,visibility=measured,reachable=reachable,
            union=visibility.union(measured,reachable),q_plan=z['q'],direction=z['direction'],
            scan_time=z['scan_time'],ptp_pi=z['ptp_pi'],from_home_pi=z['from_home_pi'],to_home_pi=z['to_home_pi'])


def solve_nominal(fid):
    from nbv import baselines
    nominal=load_nominal(fid)
    ilp,ilp_info=baselines.ilp_tsp_plan(nominal)
    joint,joint_info=baselines.joint_opt_plan(nominal,hint=ilp,time_limit_s=300,workers=8)
    return dict(part=str(fid),n_candidates=48,ilp_plan=ilp,ilp_info=ilp_info,
                joint_plan=joint,joint_info=joint_info,
                settings=dict(set_cover_seconds_per_stage=60,cp_sat_seed=0,workers=8,
                              joint_seconds=300,coverage=.95,tsp='Held-Karp <=16; routing GLS 10 s otherwise'))


def prepare_missing():
    report=dry_run()
    allowed={'nominal_points','ilp_tsp_and_joint_plans_with_solver_info'}
    forbidden=[e['path'] for e in report['inputs'] if not e['exists'] and e['role'] not in allowed]
    if forbidden: raise ValueError(f'missing inputs not authorized for nominal-only generation: {forbidden}')
    proof=prove_point_determinism('scratch/incident1-point-determinism')
    generated=[]
    for e in report['inputs']:
        if not e['exists'] and e['role']=='nominal_points':
            fid=int(Path(e['path']).stem.split('_')[1])
            _nominal_points(fid,Path('data/cache/visibility'),parts.load_yaml('configs/parts_v2.yaml'))
            generated.append(e['path']); print('Generated nominal points:',e['path'],flush=True)
    jobs=[e for e in report['inputs'] if not e['exists'] and e['role']=='ilp_tsp_and_joint_plans_with_solver_info']
    if jobs:
        with ProcessPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(solve_nominal,int(e['part'])):e for e in jobs}
            for future in as_completed(futures):
                e=futures[future]; result=future.result()
                path=Path(e['path']); path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
                generated.append(str(path)); print('Generated nominal plan:',json.dumps(result),flush=True)
    return dict(generated=generated,point_determinism=proof,
                note='No real-variant content used; no environment, episode or test performance. Parallel CP-SAT uses frozen settings; wall-time-limited joint solutions are not claimed bitwise deterministic.')


def make_input_manifest(inventory):
    if inventory['missing']: raise ValueError(f"missing inputs: {inventory['missing']}")
    return {'version':1,'inputs':[{k:v for k,v in e.items() if k!='exists'}|{'sha256':sha256(e['path'])}
                                 for e in inventory['inputs']]}


def verify_input_manifest(manifest, inventory):
    if inventory['missing']: raise ValueError('missing inputs')
    if manifest.get('version')!=1 or {e['path'] for e in manifest['inputs']}!={e['path'] for e in inventory['inputs']}:
        raise ValueError('input manifest path set mismatch')
    for e in manifest['inputs']:
        if sha256(e['path'])!=e['sha256']: raise ValueError(f"test input SHA-256 mismatch: {e['path']}")
