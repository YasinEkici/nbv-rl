"""D-069 recorded-pass replay, never learned or baseline policy evaluation."""
import csv
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
from nbv import evaluation,parts,robot
from nbv.env import ENV_VERSION,NBVEnv,evaluate_sequence
from nbv.experiment import eligible_cache_paths

STEP_FIELDS=('pass','t_move','t_scan','t_home','joint_motion','total_joint_motion','cycle_time',
             'coverage','coverage_nsc','coverage_raw','nominal_coverage','repair')
SUMMARY_FIELDS=('cycle_time','normalized_cycle_time','success','censored','passes_executed','passes_to_95',
                'repair_passes','total_joint_motion','coverage_at_5','coverage_at_10','nsc_at_5','nsc_at_10',
                'raw_at_5','raw_at_10','auc_coverage_time')
BASELINES=('greedy_cost','mcts','ilp_tsp','joint_opt')
AUC_DEFINITION='Primary-coverage step-function integral on [0,180 s] / 180; jumps after move+scan, constant through home. One scalar consistency check; does not determine final coverage.'
EQUALITY='Full-precision JSON decimal parsed to IEEE-754 binary64 and compared by packed bytes; exact integer/bool/null/sequence structure. No tolerances or rounded registry display cells.'


class ReplayMismatch(ValueError): pass


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_exact(actual,expected,field):
    if isinstance(expected,bool): equal=isinstance(actual,(bool,np.bool_)) and bool(actual)==expected
    elif isinstance(expected,float): equal=struct.pack('<d',float(actual))==struct.pack('<d',expected)
    elif isinstance(expected,(list,tuple)):
        equal=len(actual)==len(expected)
        if equal:
            for i,(a,e) in enumerate(zip(actual,expected)): require_exact(a,e,f'{field}[{i}]')
    else: equal=actual==expected
    if not equal: raise ReplayMismatch(f'{field}: recorded={expected!r}, replay={actual!r}')


def require_allowed(variant,manifest):
    matches=[v for v in manifest['variants'] if v['variant_id']==variant]
    if len(matches)!=1 or matches[0]['split'] not in ('train','val'):
        raise ValueError('real test holdout is closed; replay requires explicit train/VAL identity')
    return matches[0]['split']


def source_records(row,outputs='outputs'):
    if row['part_split'] not in ('val','train'): raise ValueError('closed test output access refused')
    path=Path(outputs)/row['run_id']/'episodes.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def replay_episode(env,stored,reference,*,stage):
    if stage not in ('A','B'): raise ValueError('unknown oracle stage')
    sequence=list(stored['sequence'])
    flags=stored.get('repair',[False]*len(sequence))
    if len(flags)!=len(sequence) or sum(flags)!=stored['repair_passes']:
        raise ReplayMismatch('recorded repair flags/count unavailable or inconsistent')
    prefix=next((i for i,flag in enumerate(flags) if flag),len(flags))
    if any(not flag for flag in flags[prefix:]): raise ReplayMismatch('non-contiguous recorded repair phase')
    trace=[]; current=env.q_home.copy(); original=env.step
    def traced(action):
        nonlocal current
        move,sweep=robot.transfer(current,env.nominal.q_plan[int(action)],env.qlim,env.v,env.a)
        obs,reward,done,truncated,info=original(action)
        require_exact(move,info['agent']['t_move'],'transfer_time')
        if not np.array_equal(obs['q'],sweep[-1]): raise ReplayMismatch('executed q differs from original transfer routine')
        trace.append(dict(q_start=current.tolist(),q_end=obs['q'].tolist(),q_samples=sweep.tolist(),
                          elapsed_before_home=float(env._t_move+env._t_scan)))
        current=obs['q'].copy()
        return obs,reward,done,truncated,info
    env.step=traced
    try:
        remaining=iter(sequence[prefix:])
        # This iterator only supplies recorded repair passes; it makes no policy decisions.
        repair=(lambda *_:next(remaining,None)) if prefix<len(sequence) else None
        result=evaluate_sequence(env,stored['variant'],sequence[:prefix],repair)
    finally: env.step=original
    fresh={'sequence':[s['pass'] for s in result['steps']], 'censored':result['censored'],
           'normalized_cycle_time':result['cycle_time']/reference,**evaluation.episode_metrics(result)}
    require_exact(fresh['sequence'],sequence,'sequence')
    for key in SUMMARY_FIELDS:
        if key not in stored: raise ReplayMismatch(f'missing recorded summary {key}')
        require_exact(fresh[key],stored[key],key)
    if stage=='A':
        if len(stored['steps'])!=len(result['steps']): raise ReplayMismatch('step count')
        for i,(a,e) in enumerate(zip(result['steps'],stored['steps'])):
            for key in STEP_FIELDS: require_exact(a[key],e[key],f'steps[{i}].{key}')
    verification=dict(stage=stage,oracle='full recorded steps' if stage=='A' else 'partial recorded summaries',
        exact=True,matched_fields=['sequence',*SUMMARY_FIELDS],matched_step_fields=list(STEP_FIELDS) if stage=='A' else [],
        equality=EQUALITY,auc_definition=AUC_DEFINITION,
        final_coverage_status='derived from replay; exact full-step oracle' if stage=='A' else 'derived from replay; consistent with partial oracle',
        final_coverage_label_tr='kayitli adimlarla birebir' if stage=='A' else 'yeniden oynatmadan türetildi; kısmi oracle ile tutarlı')
    return dict(variant=stored['variant'],part=stored['part'],summary=fresh,steps=result['steps'],joint_trace=trace,
        final_coverage=result['steps'][-1]['coverage'] if result['steps'] else 0.,verification=verification)


def validate_all(registry='results/registry.csv',outputs='outputs'):
    manifest=parts.load_yaml('configs/split_v1.yaml')
    variants=[v['variant_id'] for v in manifest['variants'] if v['split']=='val']
    for vid in variants: require_allowed(vid,manifest)
    rows=list(csv.DictReader(open(registry,encoding='utf-8')))
    baseline=[r for r in rows if r['status']=='canonical' and r['plan']=='001' and r['part_split']=='val'
              and r['method'] in BASELINES and r['env_version']==ENV_VERSION and json.loads(r['extra_json']).get('n_candidates')==48]
    learners=[r for r in rows if r['status']=='canonical' and r['part_split']=='val' and json.loads(r['extra_json']).get('tag')=='p002']
    if len(baseline)!=13 or len(learners)!=60: raise ValueError('exact canonical VAL source set required')
    ilp=next(r for r in baseline if r['method']=='ilp_tsp')
    reference={e['variant']:e['cycle_time'] for e in source_records(ilp,outputs)}
    if set(reference)!=set(variants): raise ValueError('VAL reference variant mismatch')
    state=evaluation.git_state()
    if state['dirty']: raise ValueError('commit replay code before validating real sources')
    provenance=state|dict(code_sha256=evaluation.code_hash(),env_version=ENV_VERSION,
        caches=evaluation.selected_cache_manifest(eligible_cache_paths([],variants)),variants=variants,
        registry_sha256=digest(registry),auc_definition=AUC_DEFINITION,equality=EQUALITY)
    directory=Path(outputs)/evaluation.new_run_id('p004-val-recorded-replay')
    directory.mkdir(parents=True,exist_ok=False)
    target=directory/'episodes.jsonl'
    env=NBVEnv(variants); sources={}; counts={'A':0,'B':0}; fields={'A':list(STEP_FIELDS),'B':list(SUMMARY_FIELDS)}
    try:
        with target.open('x',encoding='utf-8') as stream:
            for stage,group in [('A',baseline),('B',learners)]:
                for row in group:
                    records=source_records(row,outputs)
                    if len(records)!=15 or {r['variant'] for r in records}!=set(variants): raise ReplayMismatch('source variant set')
                    path=Path(outputs)/row['run_id']/'episodes.jsonl'
                    sources[row['run_id']]=dict(sha256=digest(path),commit=row['commit'],method=row['method'],seed=row['seed'])
                    extra=json.loads(row['extra_json'])
                    label=row['method'] if stage=='A' else f"{row['method']}/{extra['phi']}"
                    for stored in records:
                        require_allowed(stored['variant'],manifest)
                        fresh=replay_episode(env,stored,reference[stored['variant']],stage=stage)
                        fresh.update(source_run=row['run_id'],method=label,seed=int(row['seed'] or 0),split='val')
                        stream.write(json.dumps(fresh,allow_nan=False)+'\n'); counts[stage]+=1
                print(f'Exact VAL replay stage {stage}: {counts[stage]} episodes passed',flush=True)
    except Exception as exc:
        (directory/'failure.json').write_text(json.dumps({'error':str(exc),'counts':counts,'publication_blocked':True},indent=2),encoding='utf-8')
        raise
    result=provenance|dict(schema_version=1,complete=True,counts=counts,matched_fields=fields,sources=sources,
        replay_file=target.as_posix(),replay_sha256=digest(target),mismatches=[],
        learner_final_coverage='derived from replay; consistent with partial oracle',
        no_policy_calls=True,no_test_access=True)
    destination=Path('results/visualization'); destination.mkdir(parents=True,exist_ok=True)
    with (destination/'replay_validation_v1.json').open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)
    return result


def validated_records(path='results/visualization/replay_validation_v1.json'):
    manifest=json.loads(Path(path).read_text(encoding='utf-8'))
    if (not manifest['complete'] or manifest['mismatches'] or manifest['counts']!={'A':195,'B':900}
            or digest(manifest['replay_file'])!=manifest['replay_sha256']): raise ValueError('unvalidated replay artifact')
    records=[json.loads(line) for line in Path(manifest['replay_file']).read_text(encoding='utf-8').splitlines()]
    if len(records)!=1095 or any(r['split']!='val' or not r['verification']['exact'] for r in records): raise ValueError('invalid replay contents')
    return records,manifest
