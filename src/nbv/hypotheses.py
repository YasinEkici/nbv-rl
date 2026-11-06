"""Descriptive Plan 002 criteria; no significance tests or selection."""
import numpy as np
from nbv.evaluation import nested_mean, paired_difference, per_part


def _three(data):
    if len(data) != 3:
        raise ValueError('predeclared criteria require exactly three parts')


def h1(agent, greedy):
    _three(agent)
    a, g = per_part(agent), per_part(greedy)
    if a.keys() != g.keys():
        raise ValueError('part mismatch')
    thresholds = {p: (1 + g[p]) / 2 for p in a}
    meet = {p: a[p] <= thresholds[p] + 1e-12 for p in a}
    return {'supported': bool(np.mean(list(a.values())) <= np.mean(list(thresholds.values())) + 1e-12 and sum(meet.values()) >= 2),
            'thresholds': thresholds, 'parts_meeting': meet,
            'gap_closed': {p: (g[p] - a[p]) / (g[p] - 1) if abs(g[p] - 1) > 1e-12 else None for p in a}}


def forward(sarsa, myopic):
    _three(sarsa)
    if sarsa.keys() != myopic.keys() or any(sarsa[p].keys() != myopic[p].keys() for p in sarsa):
        raise ValueError('paired variants must match exactly')
    differences = paired_difference(sarsa, myopic)
    per = per_part(differences)
    mean = nested_mean(differences)
    return {'supported': bool(mean <= -.02 + 1e-12 and sum(v < 0 for v in per.values()) >= 2),
            'mean_difference': mean, 'paired_difference': per}


def shaping(curves, final):
    if curves.keys() != final.keys() or 'linear' not in final:
        raise ValueError('matching Phi arms and linear reference required')
    auc = {}
    expected_steps = None
    for name, values in curves.items():
        steps, scores = np.asarray(values, float).T
        if len(steps) < 2 or not np.isfinite(values).all() or np.any(np.diff(steps) <= 0):
            raise ValueError('finite increasing learning curves required')
        if expected_steps is not None and not np.array_equal(steps, expected_steps):
            raise ValueError('same evaluation grid required')
        expected_steps = steps
        # No invented step-zero measurement; integrate the common observed interval.
        auc[name] = float(np.trapezoid(scores, steps))
    relative = (max(auc.values()) - min(auc.values())) / min(auc.values())
    differences = {name: nested_mean(data) - nested_mean(final['linear']) for name, data in final.items()}
    speed = relative >= .1 - 1e-12
    close = all(abs(v) <= .02 + 1e-12 for v in differences.values())
    return {'supported': bool(speed and close), 'speed_changes': bool(speed), 'final_close': bool(close),
            'auc': auc, 'relative_auc_range': relative, 'final_difference': differences}
