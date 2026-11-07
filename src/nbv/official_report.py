"""Registry-driven VAL tables and exported learning curves; never opens part caches."""
import csv
import json
import os
from pathlib import Path

import numpy as np
from nbv import evaluation
from nbv.env import ENV_VERSION
from nbv.report_common import model_limitations


def summarize_episodes(records):
    parts = evaluation.per_part(evaluation.nest(records, 'normalized_cycle_time'))
    raw = evaluation.per_part(evaluation.nest(records, 'cycle_time'))
    success = evaluation.per_part(evaluation.nest(records, 'success'))
    return {'parts': parts, 'mean': float(np.mean(list(parts.values()))),
            'raw_s': float(np.mean(list(raw.values()))), 'success': float(np.mean(list(success.values())))}


def curve_summary(histories):
    steps = [h['step'] for h in histories[0]]
    if any([h['step'] for h in history] != steps for history in histories):
        raise ValueError('learning curves do not share evaluation grid')
    scores = np.asarray([[h['normalized_cycle_time'] for h in history] for history in histories])
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite learning curve')
    return {'steps': steps, 'mean': scores.mean(axis=0).tolist(),
            'low': scores.min(axis=0).tolist(), 'high': scores.max(axis=0).tolist()}


def report(registry='results/registry.csv', outputs='outputs', destination='results'):
    rows = list(csv.DictReader(open(registry, encoding='utf-8')))
    runs = [(r, json.loads(r['extra_json'])) for r in rows if r['plan'] == '002' and r['part_split'] == 'val'
            and r['env_version'] == ENV_VERSION and json.loads(r['extra_json']).get('tag') in ('p002', 'p002full')]
    groups = {}
    for row, extra in runs:
        key = (row['method'], extra['phi'], extra['feature_count'])
        groups.setdefault(key, []).append((row, extra))
    if len(runs) != 70 or len(groups) != 7 or any(len(rs) != 10 or {int(r['seed']) for r, _ in rs} != set(range(10)) for rs in groups.values()):
        raise ValueError('expected all seventy runs with ten disjoint official seeds per arm')
    dest = Path(destination)
    figure_dir = dest / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(Path('scratch/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summaries, curves, table = [], {}, []
    for (method, phi, d), rs in groups.items():
        tail, final = [], []
        for row, extra in rs:
            directory = Path(outputs) / row['run_id']
            tail += [json.loads(line) for line in (directory / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
            final += json.loads((directory / 'training.json').read_text(encoding='utf-8'))['secondary']
        main, secondary = summarize_episodes(tail), summarize_episodes(final)
        label = f'{method}/{phi}/{d}'
        summaries.append(dict(arm=label, status='canonical' if d == 11 else 'exploratory', **main,
                              final_mean=secondary['mean'], final_parts=secondary['parts'],
                              final_success=secondary['success'], diverged=sum(e['diverged'] for _, e in rs)))
        curve = curve_summary([e['history'] for _, e in rs])
        curves[label] = curve
        for part, value in main['parts'].items():
            table.append(dict(arm=label, part=part, tail_mean=value, final_weights=secondary['parts'][part],
                              success_rate=summarize_episodes([e for e in tail if str(e['part']) == part])['success']))
        fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
        ax.plot(curve['steps'], curve['mean'], label=label)
        ax.fill_between(curve['steps'], curve['low'], curve['high'], alpha=.2, label='Seed min–max (0–9)')
        ax.set(xlabel='Training environment steps', ylabel='VAL normalized cycle time', title=label)
        ax.legend()
        for extension in ('png', 'pdf'):
            fig.savefig(figure_dir / (label.replace('/', '_') + f'.{extension}'), dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    for label, curve in curves.items():
        if label.endswith('/16'):
            continue
        line, = ax.plot(curve['steps'], curve['mean'], label=label)
        ax.fill_between(curve['steps'], curve['low'], curve['high'], color=line.get_color(), alpha=.1)
    ax.set(xlabel='Training environment steps', ylabel='VAL normalized cycle time', title='Official VAL curves: mean and seed min–max')
    ax.legend()
    for extension in ('png', 'pdf'):
        fig.savefig(figure_dir / f'official_val_curves_v1.{extension}', dpi=180)
    plt.close(fig)
    (dest / 'val_curves_v1.json').write_text(json.dumps(curves, indent=2), encoding='utf-8')
    with (dest / 'val_summary_v1.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    part_ids = sorted(summaries[0]['parts'])
    lines = ['# Plan 002 official VAL report', '',
             'Official seeds 0–9, 2000000 steps; final-10% tail policy. Equal means over parts, variants and seeds. Censored episodes retained with success.',
             'The 16-feature arm is exploratory, VAL only, and excluded from hypotheses/test. Final weights are secondary and never selected.',
             'Alpha normalization differs between methods (D-054); Phi ablations inherit the linear-arm setting. These are limitations, not corrected by extra tuning.', '',
             '| Arm | ' + ' | '.join(part_ids) + ' | Mean | Raw seconds | Success | Diverged | Final weights (secondary) | Final success |',
             '|---|' + '---:|' * (len(part_ids) + 6)]
    for s in summaries:
        lines.append('| ' + s['arm'] + ' | ' + ' | '.join(f"{s['parts'][p]:.6f}" for p in part_ids)
                     + f" | {s['mean']:.6f} | {s['raw_s']:.3f} | {s['success']:.4f} | {s['diverged']}/10 | {s['final_mean']:.6f} | {s['final_success']:.4f} |")
    lines += ['', 'Canonical ENV v4 VAL baselines (Plan 001):', '',
              '| Baseline | ' + ' | '.join(part_ids) + ' | Mean | Success |', '|---|' + '---:|' * (len(part_ids) + 2)]
    for method in ('greedy_cost', 'mcts', 'ilp_tsp', 'joint_opt', 'random', 'fixed', 'greedy_ig'):
        eps = []
        for row in rows:
            if (row['plan'] == '001' and row['status'] == 'canonical' and row['env_version'] == ENV_VERSION
                    and row['part_split'] == 'val' and row['method'] == method and json.loads(row['extra_json']).get('n_candidates') == 48):
                eps += [json.loads(line) for line in (Path(outputs) / row['run_id'] / 'episodes.jsonl').read_text().splitlines()]
        summary = summarize_episodes(eps)
        lines.append('| ' + method + ' | ' + ' | '.join(f"{summary['parts'][p]:.6f}" for p in part_ids)
                     + f" | {summary['mean']:.6f} | {summary['success']:.4f} |")
    timing = [json.loads(r['extra_json']) for r in rows if json.loads(r['extra_json']).get('tag') == 'p002timing']
    if len(timing) != 1:
        raise ValueError('one completed batch timing record required')
    lines += ['', f"Measured batch wall time: {timing[0]['elapsed_s']:.2f} s ({timing[0]['elapsed_s']/3600:.3f} h), {timing[0]['workers']} workers.", '',
              '![Official arms: seed means and min–max ranges](figures/official_val_curves_v1.png)', '',
              '![Exploratory full-feature SARSA: seed mean and min–max range](figures/sarsa_linear_16.png)', '',
              'Individual arm PNG/PDF figures are in figures/. Ranges describe seeds, not confidence or bootstrap intervals.',
              'This VAL-only table makes no hypothesis verdict. The separately authorized completed test evaluation is reported in test_v1.md.']
    (dest / 'official_val_v1.md').write_text('\n'.join(lines) + model_limitations(), encoding='utf-8')
    return summaries
