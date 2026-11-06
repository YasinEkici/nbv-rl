"""Evaluation-only persistence and explicit campaign completeness; no learning."""
import csv
import json
from pathlib import Path
import numpy as np
from nbv import evaluation, parts
from nbv.env import ENV_VERSION, NBVEnv
from nbv.feature_check import CONFIG_FILES

BASELINES=('random','fixed','greedy_ig','greedy_cost','ilp_tsp','mcts','joint_opt')
METRICS=('part','variant','success','censored','normalized_cycle_time','cycle_time','passes_to_95',
         'coverage_at_5','coverage_at_10','auc_coverage_time','total_joint_motion','repair_passes',
         'nsc_at_5','nsc_at_10','raw_at_5','raw_at_10','success_without_repair')


def campaign_complete(keys):
    from nbv.test_evaluation import ARMS
    expected={(m,None,'baseline',s) for m in BASELINES for s in (range(10) if m in ('random','mcts') else [0])}
    expected|={(m,p,k,s) for m,p in ARMS for k in ('tail','final') for s in range(10)}
    if len(keys)!=145 or set(keys)!=expected:
        raise ValueError('incomplete or duplicate test evaluation campaign')
    return len(keys)


def save_evaluation(method,seed,records,params,extra,split,provenance,registry,outputs,tag,budget='n/a'):
    for e in records:
        if not all(np.isfinite(e[k]) for k in ('normalized_cycle_time','cycle_time')):
            raise ValueError('nonfinite frozen evaluation')
    run_id=evaluation.new_run_id(f'{tag}-{method}-{split}-s{seed}')
    params=params|{'run_id':run_id}
    records=[e|{'method':method,'seed':seed,'split':split,'run_id':run_id,
                'success_without_repair':e.get('planner',{}).get('success_without_repair')} for e in records]
    summary=evaluation.aggregate(records)
    evaluation.write_run(Path(outputs)/run_id,params,provenance,records,summary)
    extra=extra|{'tag':tag,'n_candidates':48,'normalized_cycle_time':summary['normalized_cycle_time'],
                 'episode_metrics':[{k:e.get(k) for k in METRICS}|{'seed':seed,'method':method} for e in records]}
    if 'source_run' in params: extra['source_run']=params['source_run']
    row=evaluation.registry_row(run_id,method,evaluation.config_ref(provenance['config_files'],params),split,seed,
        summary,extra,ENV_VERSION,'v1',evaluation.commit_label(provenance),status='canonical',plan='002',budget_steps=budget)
    evaluation.append_registry(Path(registry),[row])
    return row


def evaluate_train(registry='results/registry.csv',outputs='outputs'):
    from nbv.test_evaluation import preflight, artifact_hashes
    from nbv.experiment import eligible_cache_paths
    rows=list(csv.DictReader(open(registry,encoding='utf-8')))
    if any(json.loads(r['extra_json']).get('tag')=='p002train' for r in rows):
        raise ValueError('train evaluation already recorded; refusing duplicates')
    official,state=preflight(understand=True,registry=registry,outputs=outputs)
    variants=[v['variant_id'] for v in parts.load_yaml('configs/split_v1.yaml')['variants'] if v['split']=='train']
    refs=[r for r in rows if r['plan']=='001' and r['status']=='canonical' and r['env_version']==ENV_VERSION
          and r['part_split']=='train' and r['method']=='ilp_tsp' and json.loads(r['extra_json']).get('n_candidates')==48]
    if len(refs)!=1: raise ValueError('one canonical train ILP reference required')
    refpath=Path(outputs)/refs[0]['run_id']/'episodes.jsonl'
    reference={e['variant']:e['cycle_time'] for e in map(json.loads,refpath.read_text(encoding='utf-8').splitlines())}
    if set(reference)!=set(variants): raise ValueError('train reference mismatch')
    provenance=state|{'code_sha256':evaluation.code_hash(),'env_version':ENV_VERSION,'split_version':'v1',
                      'config_files':evaluation.file_hashes(CONFIG_FILES),
                      'caches':evaluation.selected_cache_manifest(eligible_cache_paths(variants,[])),
                      'reference':evaluation.file_hashes([str(refpath)]),'evaluation_variants':variants}
    env,cache=NBVEnv(variants),{}
    for index,row in enumerate(official):
        extra=json.loads(row['extra_json'])
        directory=Path(outputs)/row['run_id']
        with np.load(directory/'weights.npz') as weights:
            for kind in ('tail','final'):
                records=evaluation.evaluate_weights(env,variants,weights[kind],reference,reduced=True,feature_cache=cache)
                params={'source_run':row['run_id'],'weight_kind':kind,'phi':extra['phi'],'features':11}
                save_evaluation(row['method'],int(row['seed']),records,params,
                    {'weight_kind':kind,'phi':extra['phi'],'source_artifacts':artifact_hashes(directory)},
                    'train',provenance,registry,outputs,'p002train',2000000)
        if (index+1)%10==0: print(f'Train evaluation: {index+1}/60 checkpoints, both weight kinds',flush=True)
