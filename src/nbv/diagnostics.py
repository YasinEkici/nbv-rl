"""Exploratory checks of saved tuning policies; never selects new settings."""

import csv
import itertools
import json
from pathlib import Path

import numpy as np

from nbv import baselines, evaluation, shaping
from nbv.agents.linear import EpsilonSchedule
from nbv.agents.train import train
from nbv.env import ENV_VERSION, NBVEnv, evaluate_policy
from nbv.experiment import context, make_agent


def sequence_overlap(a, b):
    """Position-wise matches / longer length; unmatched suffixes count as disagreements."""
    return sum(x == y for x, y in zip(a, b)) / max(len(a), len(b), 1)


def feature_statistics(samples):
    x = np.asarray(samples, float)
    gram = x.T @ x / len(x)  # empirical D = I/n, repeated observations retain occupancy
    rank = int(np.linalg.matrix_rank(gram))
    # X_max bounds both matrix infinity norms (not the largest individual entry).
    xmax = float(max(np.linalg.norm(x, np.inf), np.linalg.norm(x.T, np.inf)))
    threshold = float(np.linalg.norm(np.linalg.inv(gram), np.inf) * xmax**2 - 1) if rank == x.shape[1] else None
    return {'n': len(x), 'rank': rank, 'dimension': x.shape[1],
            'mean_squared_norm': float(np.trace(gram)), 'gram_eigenvalues': np.linalg.eigvalsh(gram).tolist(),
            'x_max_matrix_bound': xmax, 'lemma_threshold': threshold,
            'gram': gram.tolist()}


def run(registry='results/registry.csv', outputs='outputs'):
    ctx = context('exploratory', registry, outputs)
    rows = list(csv.DictReader(open(registry, encoding='utf-8')))
    tuning = [(r, json.loads(r['extra_json'])) for r in rows if r['method'] == 'regq'
              and r['part_split'] == 'val' and json.loads(r['extra_json']).get('tag') == 'p002t']
    if len(tuning) != 36:
        raise ValueError('expected exactly 36 saved RegQ tuning runs')
    env, policies, weights, checks = NBVEnv(ctx['val']), {}, {}, []
    greedy = {vid: [s['pass'] for s in evaluate_policy(env, vid,
              lambda e: baselines.greedy_cost_policy(e.v, e.a))['steps']] for vid in ctx['val']}
    for row, extra in tuning:
        with np.load(Path(outputs) / row['run_id'] / 'weights.npz') as saved:
            w = saved['tail'].copy()
        records = evaluation.evaluate_weights(env, ctx['val'], w, ctx['reference'], reduced=True)
        stored = [json.loads(line) for line in (Path(outputs) / row['run_id'] / 'episodes.jsonl').read_text().splitlines()]
        if [r['sequence'] for r in records] != [r['sequence'] for r in stored]:
            raise RuntimeError('STOP: saved tail weights do not reproduce recorded policy')
        if not np.isclose(evaluation.aggregate(records)['normalized_cycle_time'],
                          evaluation.aggregate(stored)['normalized_cycle_time'], atol=1e-12, rtol=0):
            raise RuntimeError('STOP: saved tail weights do not reproduce recorded score')
        key = (extra['grid_id'], int(row['seed']))
        weights[key], policies[key] = w, {r['variant']: r['sequence'] for r in records}
        checks.append({'grid_id': key[0], 'seed': key[1], 'setting': extra['setting'], 'weights': w.tolist(),
                       'norm': float(np.linalg.norm(w)), 'score': evaluation.aggregate(records)['normalized_cycle_time'],
                       'greedy_overlap': float(np.mean([sequence_overlap(r['sequence'], greedy[r['variant']]) for r in records]))})
    pairs = []
    for a, b in itertools.combinations(checks, 2):
        sa, sb = a['setting'], b['setting']
        if sa['eta'] != sb['eta'] or sa['epsilon'] != sb['epsilon']:
            continue
        ka, kb = (a['grid_id'], a['seed']), (b['grid_id'], b['seed'])
        pairs.append({'a': list(ka), 'b': list(kb), 'eta': sa['eta'], 'epsilon': sa['epsilon'],
                      'weight_difference': float(np.linalg.norm(weights[ka] - weights[kb])),
                      'overlap': float(np.mean([sequence_overlap(policies[ka][v], policies[kb][v]) for v in ctx['val']]))})
    selected = json.loads(Path('results/selected_settings_v1.json').read_text())
    stats = {}
    for method in ('sarsa', 'regq', 'myopic'):
        setting = selected[method]['setting']
        agent = make_agent(method, 11, setting, 1000)
        wrapper = shaping.PotentialShaping(NBVEnv(ctx['train']), phi='linear', alpha=setting.get('phi_alpha', 50.))
        samples = []
        # Exact first 5000 transitions of the selected 200000-step tuning run: same seed/schedule.
        replay = train(agent, wrapper, seed=1000, budget=5000, eval_every=5000, reduced=True,
                       schedule=EpsilonSchedule.constant(.05), evaluate=lambda w: {},
                       observe_features=lambda x: samples.append(x), stop_on_nonfinite=True)
        stat = feature_statistics(samples)
        stat.update(effective_alpha=agent.alpha, normalized_step=agent.alpha * stat['mean_squared_norm'])
        stats[method] = stat
        summary = evaluation.aggregate(evaluation.evaluate_weights(env, ctx['val'], weights[('regq-01', 1000)], ctx['reference'], reduced=True))
        run_id = evaluation.new_run_id(f'p002diag-{method}-{ENV_VERSION}')
        extra = {'tag': 'p002diag', 'diagnostic_only': True, 'sample_method': method, 'feature_statistics': stat}
        if method == 'regq':
            extra.update(saved_policies=checks, comparisons=pairs, interpretation='different weights, robust argmax; saved scores reproduced')
        evaluation.write_run(Path(outputs) / run_id, {'method': method, 'setting': setting, 'budget': 5000}, ctx['provenance'], [], summary)
        evaluation.append_registry(Path(registry), [evaluation.registry_row(run_id, f'{method}_diagnostic',
            evaluation.config_ref(ctx['provenance']['config_files'], {'diagnostic': method}), 'val', 1000, summary, extra,
            ENV_VERSION, 'v1', evaluation.commit_label(ctx['provenance']), plan='002', budget_steps=5000)])
    report(registry)


def report(registry='results/registry.csv', out='results/regq_diagnostic_v1.md'):
    data = [json.loads(r['extra_json']) for r in csv.DictReader(open(registry, encoding='utf-8'))
            if json.loads(r['extra_json']).get('tag') == 'p002diag']
    regq = next(d for d in data if d['sample_method'] == 'regq')
    lines = ['# Pre-official diagnostics (exploratory)', '',
             'Saved tail weights reproduce all 36 recorded final policies and scores exactly. No new tuning or selection.',
             'Overlap = position-wise matches / longer sequence length, averaged equally over 15 VAL variants; suffixes disagree.', '',
             '| Grid | Seed | Norm | Overlap with greedy-IG/cost | Tail-mean vector (11 coordinates) |', '|---|---:|---:|---:|---|']
    for p in regq['saved_policies']:
        lines.append(f"| {p['grid_id']} | {p['seed']} | {p['norm']:.8f} | {p['greedy_overlap']:.6f} | " + ', '.join(f'{x:.6f}' for x in p['weights']) + ' |')
    lines += ['', 'Same eta/epsilon; compare alpha pairs at the same seed (seed-noise comparisons summarized below).', '',
              '| A | B | Seed | Norm of weight difference | Sequence overlap |', '|---|---|---:|---:|---:|']
    for p in regq['comparisons']:
        if p['a'][1] == p['b'][1]:
            lines.append(f"| {p['a'][0]} | {p['b'][0]} | {p['a'][1]} | {p['weight_difference']:.8f} | {p['overlap']:.6f} |")
    lines += ['', '| Method | Samples | E[‖x‖²] | Effective alpha | alpha × E[‖x‖²] | Gram rank | Threshold |', '|---|---:|---:|---:|---:|---:|---|']
    for d in data:
        s = d['feature_statistics']
        lines.append(f"| {d['sample_method']} | {s['n']} | {s['mean_squared_norm']:.6f} | {s['effective_alpha']:.9f} | {s['normalized_step']:.6f} | {s['rank']}/{s['dimension']} | {s['lemma_threshold']} |")
    pairs = regq['comparisons']
    lines += ['', f"All same-eta/epsilon comparisons: weight-difference range {min(p['weight_difference'] for p in pairs):.8f}–{max(p['weight_difference'] for p in pairs):.8f}; overlap range {min(p['overlap'] for p in pairs):.6f}–{max(p['overlap'] for p in pairs):.6f}.", '',
              'Samples replay the first 5000 actual selected-tuning transitions (seed 1000, original constant epsilon), through the existing loop. Empirical D=I/n, including repeated visits. No claim of stationary occupancy.',
              'A singular Gram matrix makes the user-specified inverse-based threshold undefined; no pseudoinverse is substituted. The current NeurIPS paper numbers contraction/uniqueness as Lemmas 3.2–3.3, unlike the cited older Lemma 3.1. Neither small eta nor identical discrete policies establishes its assumptions.',
              'Test: test_regq_effective_alpha_changes_updates_on_identical_data; alpha enters every update. Tail weights are loaded explicitly and independently reproduce the saved evaluation.',
              'This supports explanation (i), not a claim of unique weights or theoretical convergence. Alpha-scale asymmetry remains a comparison limitation (Patterson et al. 2024).']
    Path(out).write_text('\n'.join(lines) + '\n', encoding='utf-8')
