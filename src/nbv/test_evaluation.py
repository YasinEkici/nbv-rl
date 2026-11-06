"""One-time held-out evaluation. Importing this module does not read any part cache.

Only the explicitly gated CLI may execute the real evaluation, after separate user approval.
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

from nbv import evaluation
from nbv.env import ENV_VERSION

ARMS = {('sarsa', p) for p in ('linear', 'none', 'convex', 'value_estimate')} | {('regq', 'linear'), ('myopic', 'linear')}
ARTIFACTS = ('weights.npz', 'training.json', 'config.yaml', 'episodes.jsonl')


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


def checkpoint_complete(directory, identity=None):
    """Validate actual final-step artifacts, not just a declared registry budget."""
    directory = Path(directory)
    try:
        provenance = json.loads((directory / 'provenance.json').read_text(encoding='utf-8'))
        config = yaml.safe_load((directory / 'config.yaml').read_text(encoding='utf-8'))
        if identity is not None:
            config={k:identity[k] for k in ('budget','features')}|config
        training = json.loads((directory / 'training.json').read_text(encoding='utf-8'))
        if (provenance['dirty'] or provenance['env_version'] != ENV_VERSION or provenance['split_version'] != 'v1'
                or config['budget'] != 2000000 or config['features'] != 11):
            return False
        if [h['step'] for h in training['history']] != list(range(100000, 2000001, 100000)):
            return False
        if not all(np.isfinite(h['weight_norm']) and np.isfinite(h['normalized_cycle_time']) for h in training['history']):
            return False
        with np.load(directory / 'weights.npz') as weights:
            for kind, key in [('tail', 'policy_weights'), ('final', 'final_weights')]:
                if (weights[kind].shape != (11,) or not np.isfinite(weights[kind]).all()
                        or not np.array_equal(weights[kind], training[key])):
                    return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def checkpoint_identity(row, selected):
    """D-053 identity, independent of the checkpoint's own declarations."""
    extra = json.loads(row['extra_json'])
    method = row['method']
    setting = selected[method]['setting']
    if (extra.get('setting') != setting or extra.get('feature_count') != 11
            or int(row['budget_steps']) != 2000000 or (method, extra['phi']) not in ARMS
            or int(row['seed']) not in range(10)):
        raise ValueError('registry differs from the frozen selected setting')
    alpha = setting['alpha'] / 11 if method == 'sarsa' else setting['alpha']
    if extra.get('effective_alpha') != alpha:
        raise ValueError('registry effective alpha differs from D-053')
    return dict(method=method, seed=int(row['seed']), phi=extra['phi'], features=11, budget=2000000,
                setting=setting, effective_alpha=alpha, phi_scale=setting.get('phi_alpha',50.))


def checkpoint_matches(directory, row, selected):
    try:
        identity = checkpoint_identity(row, selected)
        if not checkpoint_complete(directory,identity):
            return False
        directory = Path(directory)
        config = yaml.safe_load((directory/'config.yaml').read_text(encoding='utf-8'))
        training = json.loads((directory/'training.json').read_text(encoding='utf-8'))
        def matches(data):
            keys=('method','seed','phi','features','budget','setting','effective_alpha')
            if any(data.get(k) != identity[k] for k in keys):
                return False
            if data['setting'].get('phi_alpha',50.) != identity['phi_scale']:
                return False
            # Optional redundant declarations cannot disagree with the canonical setting dict.
            for key in ('lam','eta','epsilon','phi_alpha'):
                if key in data and data[key] != identity['setting'].get(key,50. if key=='phi_alpha' else None):
                    return False
            return True
        # Legacy missing fields may only be filled from the already validated registry identity.
        bound=config|{k:identity[k] for k in ('method','seed','phi','features','budget','setting','effective_alpha') if k not in config}
        if not matches(bound):
            return False
        if 'identity' in training and not matches(training['identity']):
            return False
        # Older training.json has no identity; its original bytes are bound by the manifest.
        if any(k in training and training[k] != identity[k] for k in ('method','seed','phi','features','budget','setting','effective_alpha')):
            return False
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        return False
    return True


def artifact_hashes(directory):
    return {name: hashlib.sha256((Path(directory)/name).read_bytes()).hexdigest() for name in ARTIFACTS}


def verify_artifact_hashes(directory, expected):
    if set(expected) != set(ARTIFACTS) or artifact_hashes(directory) != expected:
        raise ValueError('pretest artifact SHA-256 mismatch')


def make_manifest(rows, outputs='outputs', settings='results/selected_settings_v1.json'):
    selected=json.loads(Path(settings).read_text(encoding='utf-8'))
    official=validate_opening(True,{'dirty':False},rows,
                             lambda row:checkpoint_matches(Path(outputs)/row['run_id'],row,selected))
    manifests={}
    for row in official:
        directory=Path(outputs)/row['run_id']
        config=yaml.safe_load((directory/'config.yaml').read_text(encoding='utf-8'))
        identity=checkpoint_identity(row,selected)
        registry_paths={'method':'method','seed':'seed','phi':'extra_json.phi','features':'extra_json.feature_count',
                        'budget':'budget_steps','setting':'extra_json.setting','effective_alpha':'extra_json.effective_alpha'}
        sources={key:{'config':key if key in config else 'absent; registry fallback',
                      'registry':registry_paths[key],
                      'selected_settings':'setting' if key in ('setting','effective_alpha') else 'predeclared arm/seed/feature/budget rule'}
                 for key in registry_paths}
        sources['phi_scale']=('setting.phi_alpha in config/registry/selected settings' if 'phi_alpha' in identity['setting']
                              else 'fixed D-053 default 50 s in the frozen implementation; matching registry/selected setting')
        manifests[row['run_id']]={'identity':identity,'sources':sources,'sha256':artifact_hashes(directory),
                                 'val_reproduction':reproduce_validation(row,outputs)}
    return {'version':1,'settings_sha256':hashlib.sha256(Path(settings).read_bytes()).hexdigest(),
            'legacy_training_identity':'Original training.json without identity is bound to validated config/registry by raw file hashes.',
            'runs':manifests}


def reproduce_validation(row, outputs='outputs', shared=None):
    """Bind both original weight arrays to their recorded VAL trajectories, without learning."""
    from nbv import parts
    from nbv.env import NBVEnv
    from nbv.feature_check import validation_reference
    if shared is None:
        variants=[v['variant_id'] for v in parts.load_yaml('configs/split_v1.yaml')['variants'] if v['split']=='val']
        reference,_=validation_reference('results/registry.csv',outputs)
        shared=(NBVEnv(variants),variants,reference,{})
    env,variants,reference,cache=shared
    directory=Path(outputs)/row['run_id']
    training=json.loads((directory/'training.json').read_text(encoding='utf-8'))
    tail=[json.loads(line) for line in (directory/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    with np.load(directory/'weights.npz') as weights:
        for kind,stored in [('tail',tail),('final',training['secondary'])]:
            fresh=evaluation.evaluate_weights(env,variants,weights[kind],reference,reduced=True,feature_cache=cache)
            assert_val_reproduction(fresh,stored)
    return {'tail':'exact sequences, cycle times and success','final':'exact sequences, cycle times and success',
            'episodes_per_weight_kind':len(variants)}


def assert_val_reproduction(fresh, stored):
    if len(fresh)!=len(stored) or {e['variant'] for e in fresh}!={e['variant'] for e in stored}:
        raise ValueError('VAL reproduction variant mismatch')
    by_variant={e['variant']:e for e in stored}
    for e in fresh:
        old=by_variant[e['variant']]
        if any(e[key]!=old[key] for key in ('sequence','cycle_time','success')):
            raise ValueError('VAL reproduction trajectory mismatch')


def verify_manifest(manifest, official, outputs='outputs', settings='results/selected_settings_v1.json'):
    selected=json.loads(Path(settings).read_text(encoding='utf-8'))
    if (manifest.get('version') != 1 or manifest['settings_sha256'] != hashlib.sha256(Path(settings).read_bytes()).hexdigest()
            or set(manifest['runs']) != {r['run_id'] for r in official}):
        raise ValueError('pretest manifest setting or run set mismatch')
    for row in official:
        expected=manifest['runs'][row['run_id']]
        if expected['identity'] != checkpoint_identity(row,selected):
            raise ValueError('pretest manifest identity mismatch')
        if expected.get('val_reproduction',{}).get('episodes_per_weight_kind') != 15:
            raise ValueError('pretest manifest missing VAL reproduction evidence')
        verify_artifact_hashes(Path(outputs)/row['run_id'],expected['sha256'])


def preflight(*, understand=False, registry='results/registry.csv', outputs='outputs',
              manifest_path='results/pretest_manifest_v1.json', settings='results/selected_settings_v1.json'):
    state = evaluation.git_state()
    rows = load_rows(registry)
    selected_settings=json.loads(Path(settings).read_text(encoding='utf-8'))
    def valid_weights(row):
        return checkpoint_matches(Path(outputs)/row['run_id'],row,selected_settings)
    selected = validate_opening(understand, state, rows, valid_weights)
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    verify_manifest(manifest,selected,outputs,settings)
    evaluation._git('ls-files','--error-unmatch',str(manifest_path))
    state=state|{'pretest_manifest_sha256':hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()}
    return selected,state


def open_test(*, understand=False, registry='results/registry.csv', outputs='outputs'):
    selected,state=preflight(understand=understand,registry=registry,outputs=outputs)
    # Claim before any test cache is opened: even an interrupted partial evaluation cannot be rerun.
    claim_opening(Path(outputs) / 'p002-test-opened.json', state | {'official_runs': [r['run_id'] for r in selected]})
    return execute(selected, state, registry, outputs)


def prepare_joint(nominal):
    """D-035: ILP hint and unchanged solver settings; retain the reference bound."""
    from nbv import baselines
    hint, ilp_info = baselines.ilp_tsp_plan(nominal)
    plan, info = baselines.joint_opt_plan(nominal, hint=hint, time_limit_s=300, workers=8)
    return plan, info | {'ilp_hint': hint, 'ilp_info': ilp_info}


def execute(selected, state, registry, outputs):
    """Real test access lives entirely behind validate_opening + exclusive claim. NOT run in Slice 5a."""
    from nbv import baselines, parts
    from nbv.env import NBVEnv
    from nbv.experiment import eligible_cache_paths
    from nbv.feature_check import CONFIG_FILES
    from nbv.frozen_evaluation import save_evaluation, campaign_complete
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
    joint, joint_info = {}, {}
    for vid in variants:
        fid = vid.split('_')[0]
        if fid not in joint:
            env.reset(options={'variant': vid})
            joint[fid], joint_info[fid] = prepare_joint(env.nominal)
    plans_path = Path(outputs) / 'p002-test-joint-plans.json'
    with plans_path.open('x', encoding='utf-8') as stream:
        json.dump(joint, stream)
    info_path = Path(outputs) / 'p002-test-joint-info.json'
    with info_path.open('x', encoding='utf-8') as stream:
        json.dump(joint_info, stream, indent=2)
    provenance['joint_plans'] = evaluation.file_hashes([str(plans_path), str(info_path)])
    completed = []
    for method in entry.CANONICAL:
        for seed in (range(10) if method in entry.STOCHASTIC else [0]):
            task = dict(method=method, n=48, split='test', seed=seed, variants=variants,
                        run_id=evaluation.new_run_id(f'p002test-{method}-s{seed}'),
                        d_plan='results/d_plan_v2.csv', joint_plans=str(plans_path))
            completed.append(entry.run_task(task))
    reference = {r['variant']: r['cycle_time'] for o in completed if o['task']['method'] == 'ilp_tsp' for r in o['episodes']}

    keys, written = [], []
    def save(method, seed, records, params, extra, budget='n/a'):
        row=save_evaluation(method,seed,records,params,extra,'test',provenance,registry,outputs,'p002test',budget)
        keys.append((method,extra.get('phi'),extra.get('weight_kind','baseline'),seed))
        written.append(row['run_id'])
    for result in completed:
        task = result['task']
        records = [r | {'normalized_cycle_time': r['cycle_time'] / reference[r['variant']]} for r in result['episodes']]
        if task['method'] == 'joint_opt':
            for record in records:
                record['planner'].update(joint_info[str(record['part'])])
        save(task['method'], task['seed'], records, task, {'baseline': True})
    feature_cache = {}
    for index,row in enumerate(selected):
        extra = json.loads(row['extra_json'])
        with np.load(Path(outputs) / row['run_id'] / 'weights.npz') as weights:
            for kind in ('tail', 'final'):
                records = evaluation.evaluate_weights(env, variants, weights[kind], reference, reduced=True, feature_cache=feature_cache)
                save(row['method'], int(row['seed']), records,
                     {'source_run': row['run_id'], 'weight_kind': kind, 'phi': extra['phi']},
                     {'weight_kind': kind, 'phi': extra['phi'], 'source_run': row['run_id'],
                      'source_artifacts':artifact_hashes(Path(outputs)/row['run_id'])}, 2000000)
        if (index+1)%10==0: print(f'Test evaluation: {index+1}/60 checkpoints, both weight kinds',flush=True)
    campaign_complete(keys)
    completion=state|{'evaluation_runs':written,'run_count':len(written),'variants':variants,
                      'opening_claim_sha256':hashlib.sha256((Path(outputs)/'p002-test-opened.json').read_bytes()).hexdigest()}
    with (Path(outputs)/'p002-test-completed.json').open('x',encoding='utf-8') as stream:
        json.dump(completion,stream,indent=2)
    print(f'Completed one test opening: {len(written)} evaluation rows, {len(variants)} variants.',flush=True)
    return completion
