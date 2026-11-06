"""Authorized Slice 5a train/VAL tasks, fixed settings and stop rules."""
import csv
import json
from pathlib import Path
import time

from nbv.experiment import StopRun, run_tasks


def official_tasks(selected):
    arms = [('sarsa', 'linear', 11), ('regq', 'linear', 11), ('myopic', 'linear', 11),
            ('sarsa', 'none', 11), ('sarsa', 'convex', 11), ('sarsa', 'value_estimate', 11),
            ('sarsa', 'linear', 16)]
    return [dict(method=m, phi=p, features=d, setting=selected[m]['setting'], seed=s, budget=2000000,
                 _status='canonical' if d == 11 else 'exploratory', _tag='p002' if d == 11 else 'p002full',
                 _progress_path=f'scratch/official_progress/{m}_{p}_{d}_{s}.json')
            for m, p, d in arms for s in range(10)]


def check_completed(results):
    groups = {}
    for r in results:
        t = r['task']
        key = (t['method'], t['phi'], t['features'])
        groups[key] = groups.get(key, 0) + int(r['diverged'])
        if groups[key] > 5:
            raise StopRun(f'more than half the seeds diverged in arm {key}')


def run(workers=20):
    rows = list(csv.DictReader(open('results/registry.csv', encoding='utf-8')))
    if any(json.loads(r['extra_json']).get('tag') in ('p002', 'p002full') for r in rows):
        raise StopRun('official rows already exist; refusing duplicate launch')
    selected = json.loads(Path('results/selected_settings_v1.json').read_text())
    started = time.perf_counter()
    run_tasks(official_tasks(selected), status='canonical', tag='p002', workers=workers,
              max_wall_s=6 * 3600, completion_check=check_completed)
    elapsed = time.perf_counter() - started
    # Result timing itself is a registry record, so reporting never relies on a copied number.
    timing = {'tag': 'p002timing', 'elapsed_s': elapsed, 'workers': workers, 'completed': 70}
    from nbv import evaluation
    from nbv.env import ENV_VERSION
    summary = {k: float('nan') for k in ('success_rate', 'cycle_time_to_95_s', 'passes_to_95', 'coverage_at_5', 'coverage_at_10', 'auc_coverage_time')}
    evaluation.append_registry(Path('results/registry.csv'), [evaluation.registry_row(
        evaluation.new_run_id('p002timing'), 'batch_timing', 'selected_settings_v1.json', 'val', '', summary,
        timing, ENV_VERSION, 'v1', evaluation.commit_label(evaluation.git_state()), plan='002', budget_steps=140000000)])
    print(f'70 runs completed, elapsed {elapsed:.2f}s ({elapsed/3600:.3f}h)', flush=True)
