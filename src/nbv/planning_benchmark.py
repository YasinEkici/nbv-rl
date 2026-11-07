"""Exploratory deployed-policy latency on VAL only; no test-part interface or quality selection."""
import csv
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
from nbv import baselines,evaluation,parts
from nbv.agents.frozen_policy import FrozenLinearPolicy
from nbv.env import NBVEnv,ENV_VERSION
from nbv.feature_check import CONFIG_FILES
from nbv.experiment import eligible_cache_paths
from nbv.diagnostics import diagnostic_only_summary


def require_validation(variants,manifest):
    allowed={v['variant_id'] for v in manifest['variants'] if v['split']=='val'}
    if not variants or not set(variants)<=allowed: raise ValueError('planning benchmark is VAL only')
    return list(variants)


def measure_calls(policy,states,repeats=3):
    elapsed=[]
    for _ in range(repeats):
        for obs,nominal,mask in states:
            start=time.perf_counter_ns()
            policy(obs,nominal,mask)
            elapsed.append((time.perf_counter_ns()-start)/1e6)
    return dict(decisions=len(elapsed),planning_time_ms_per_decision=float(np.mean(elapsed)),
                median_ms=float(np.median(elapsed)),p95_ms=float(np.quantile(elapsed,.95)))


def benchmark(registry='results/registry.csv',outputs='outputs',repeats=3):
    rows=list(csv.DictReader(open(registry,encoding='utf-8')))
    if any(json.loads(r['extra_json']).get('tag')=='p002planning' for r in rows):
        raise ValueError('planning latency already recorded; no automatic remeasurement')
    manifest=parts.load_yaml('configs/split_v1.yaml')
    variants=require_validation([v['variant_id'] for v in manifest['variants'] if v['split']=='val'],manifest)
    state=evaluation.git_state()
    if state['dirty']: raise ValueError('commit benchmark code before timing')
    env=NBVEnv(variants)
    states=[]
    reference=baselines.greedy_cost_policy(env.v,env.a)
    for vid in variants:
        obs,_=env.reset(options={'variant':vid})
        while True:
            mask=env.action_masks().copy()
            states.append(({k:v.copy() for k,v in obs.items()},env.nominal,mask))
            action=reference(obs,env.nominal,mask)
            if action is None: break
            obs,_,terminated,truncated,_=env.step(action)
            if terminated or truncated: break
    policies={}; sources={}
    for method,phi in [('sarsa','value_estimate'),('sarsa','linear'),('regq','linear')]:
        row=next(r for r in rows if r['status']=='canonical' and r['method']==method and r['seed']=='0'
                 and json.loads(r['extra_json']).get('tag')=='p002' and json.loads(r['extra_json']).get('phi')==phi)
        label=f'{method}/{phi}'
        with np.load(Path(outputs)/row['run_id']/'weights.npz') as weights:
            policies[label]=FrozenLinearPolicy(weights['tail'],env.k_max,env.target)
        sources[label]=row['run_id']
    policies['mcts']=baselines.make_policy('mcts',env,np.random.default_rng([0,48,7]))
    policies['greedy_cost']=reference
    # Warm nominal feature preprocessing and model caches outside the measured calls.
    for policy in policies.values():
        for nominal in {id(s[1]):s for s in states}.values(): policy(*nominal)
    timings={label:measure_calls(policy,states,repeats) for label,policy in policies.items()}
    provenance=state|dict(code_sha256=evaluation.code_hash(),config_files=evaluation.file_hashes(CONFIG_FILES),
        caches=evaluation.selected_cache_manifest(eligible_cache_paths([],variants)),env_version=ENV_VERSION,split_version='v1')
    extra=dict(tag='p002planning',diagnostic_only=True,metric='planning_time_ms_per_decision',timings=timings,
        source_runs=sources,variants=variants,states=len(states),repeats=repeats,split='val',seed=0,
        processor=platform.processor(),threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        definition='Serial warm-cache inference calls on the SAME greedy-cost VAL states; feature evaluation/argmax or nominal MCTS included; environment steps, model preprocessing, cache loading and Phi computation excluded. Phi is training-only for frozen policies. Fixed official seed 0 checkpoints, no quality selection.')
    run_id=evaluation.new_run_id('p002planning-val')
    params={'split':'val','repeats':repeats,'source_runs':sources,'run_id':run_id}
    evaluation.write_run(Path(outputs)/run_id,params,provenance,[],diagnostic_only_summary())
    (Path(outputs)/run_id/'timing.json').write_text(json.dumps(extra,indent=2),encoding='utf-8')
    row=evaluation.registry_row(run_id,'planning_timing',evaluation.config_ref(provenance['config_files'],params),
        'val',0,diagnostic_only_summary(),extra,ENV_VERSION,'v1',state['commit'],status='exploratory',plan='002')
    evaluation.append_registry(Path(registry),[row])
    print(json.dumps(extra,indent=2),flush=True)
    return extra
