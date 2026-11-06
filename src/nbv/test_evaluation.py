"""One-time held-out evaluation. Importing this module does not read any part cache.

Only the explicitly gated CLI may execute the real evaluation, after separate user approval.
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

from nbv import evaluation
from nbv.env import ENV_VERSION

ARMS = {('sarsa', p) for p in ('linear', 'none', 'convex', 'value_estimate')} | {('regq', 'linear'), ('myopic', 'linear')}


def load_rows(path):
    return list(csv.DictReader(open(path, encoding='utf-8')))


def validate_opening(understand, state, rows, weight_exists):
    if not understand:
        raise ValueError('requires --i-understand-this-opens-test and separate user approval')
    if state['dirty']:
        raise ValueError('test evaluation requires a clean working tree')
    if any(r.get('part_split') == 'test' for r in rows):
        raise ValueError('test rows already exist: refusing adaptive holdout reuse')
    selected = [r for r in rows if r.get('plan') == '002' and r.get('status') == 'canonical'
                and r.get('part_split') == 'val' and json.loads(r['extra_json']).get('tag') == 'p002']
    keys = []
    for row in selected:
        extra = json.loads(row['extra_json'])
        if (row['env_version'] != ENV_VERSION or row['split_version'] != 'v1'
                or int(row['budget_steps']) != 2000000 or extra['feature_count'] != 11
                or not weight_exists(row)):
            raise ValueError('official inputs or weights incomplete')
        keys.append((row['method'], extra['phi'], int(row['seed'])))
    expected = {(m, p, s) for m, p in ARMS for s in range(10)}
    if len(keys) != 60 or set(keys) != expected:
        raise ValueError('need all sixty official runs exactly once; exploratory arm excluded')
    return selected


def claim_opening(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(state, stream, indent=2)


def open_test(*, understand=False, registry='results/registry.csv', outputs='outputs'):
    state = evaluation.git_state()
    rows = load_rows(registry)
    def valid_weights(row):
        directory = Path(outputs) / row['run_id']
        if not (directory / 'weights.npz').exists() or not (directory / 'provenance.json').exists():
            return False
        provenance = json.loads((directory / 'provenance.json').read_text())
        if provenance['dirty']:
            return False
        with np.load(directory / 'weights.npz') as weights:
            return all(weights[k].shape == (11,) and np.isfinite(weights[k]).all() for k in ('tail', 'final'))
    selected = validate_opening(understand, state, rows, valid_weights)
    # Claim before any test cache is opened: even an interrupted partial evaluation cannot be rerun.
    claim_opening(Path(outputs) / 'p002-test-opened.json', state | {'official_runs': [r['run_id'] for r in selected]})
    return execute(selected, state, registry, outputs)


def execute(selected, state, registry, outputs):
    """Real test access lives entirely behind validate_opening + exclusive claim. NOT run in Slice 5a."""
    from nbv import baselines, parts
    from nbv.env import NBVEnv
    from nbv.experiment import eligible_cache_paths
    from nbv.feature_check import CONFIG_FILES
    spec = importlib.util.spec_from_file_location('baseline_entry', Path(__file__).resolve().parents[2] / 'scripts/run_baselines.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    manifest = parts.load_yaml('configs/split_v1.yaml')
    variants = [v['variant_id'] for v in manifest['variants'] if v['split'] == 'test']
    provenance = state | {'code_sha256': evaluation.code_hash(), 'env_version': ENV_VERSION, 'split_version': 'v1',
                          'config_files': evaluation.file_hashes(CONFIG_FILES),
                          'caches': evaluation.selected_cache_manifest(eligible_cache_paths([], variants)),
                          'test_variants': variants}
    # Prepare nominal-only joint plans once inside this one-time opening, using D-035 settings.
    env = NBVEnv(variants)
    joint = {}
    for vid in variants:
        fid = vid.split('_')[0]
        if fid not in joint:
            env.reset(options={'variant': vid})
            joint[fid] = baselines.joint_opt_plan(env.nominal)[0]
    plans_path = Path(outputs) / 'p002-test-joint-plans.json'
    with plans_path.open('x', encoding='utf-8') as stream:
        json.dump(joint, stream)
    completed = []
    for method in entry.CANONICAL:
        for seed in (range(10) if method in entry.STOCHASTIC else [0]):
            task = dict(method=method, n=48, split='test', seed=seed, variants=variants,
                        run_id=evaluation.new_run_id(f'p002test-{method}-s{seed}'),
                        d_plan='results/d_plan_v2.csv', joint_plans=str(plans_path))
            completed.append(entry.run_task(task))
    reference = {r['variant']: r['cycle_time'] for o in completed if o['task']['method'] == 'ilp_tsp' for r in o['episodes']}

    def save(method, seed, records, params, extra, budget='n/a'):
        run_id = evaluation.new_run_id(f'p002test-{method}-s{seed}')
        records = [r | {'method': method, 'seed': seed, 'split': 'test', 'run_id': run_id} for r in records]
        summary = evaluation.aggregate(records)
        evaluation.write_run(Path(outputs) / run_id, params, provenance, records, summary)
        row = evaluation.registry_row(run_id, method, evaluation.config_ref(provenance['config_files'], params),
            'test', seed, summary, extra | {'tag': 'p002test', 'n_candidates': 48,
                                          'normalized_cycle_time': summary['normalized_cycle_time']},
            ENV_VERSION, 'v1', evaluation.commit_label(state), status='canonical', plan='002', budget_steps=budget)
        evaluation.append_registry(Path(registry), [row])
    for result in completed:
        task = result['task']
        records = [r | {'normalized_cycle_time': r['cycle_time'] / reference[r['variant']]} for r in result['episodes']]
        save(task['method'], task['seed'], records, task, {'baseline': True})
    feature_cache = {}
    for row in selected:
        extra = json.loads(row['extra_json'])
        with np.load(Path(outputs) / row['run_id'] / 'weights.npz') as weights:
            for kind in ('tail', 'final'):
                records = evaluation.evaluate_weights(env, variants, weights[kind], reference, reduced=True, feature_cache=feature_cache)
                save(row['method'], int(row['seed']), records,
                     {'source_run': row['run_id'], 'weight_kind': kind, 'phi': extra['phi']},
                     {'weight_kind': kind, 'phi': extra['phi'], 'source_run': row['run_id']}, 2000000)
