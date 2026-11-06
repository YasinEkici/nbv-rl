"""Predeclared descriptive thresholds; synthetic part/variant/seed values only."""
import pytest


def nested(values):
    return {str(p): {f'{p}_v0': [x, x]} for p, x in enumerate(values)}


def test_h1_requires_both_mean_threshold_and_two_of_three_parts():
    from nbv.hypotheses import h1
    greedy = nested([1.2, 1.2, 1.2])
    assert h1(nested([1.09, 1.10, 1.11]), greedy)['supported']
    assert not h1(nested([.9, 1.11, 1.11]), greedy)['supported']
    assert not h1(nested([1.0, 1.0, 1.5]), greedy)['supported']


def test_forward_averages_seeds_then_pairs_variants():
    from nbv.hypotheses import forward
    result = forward(nested([1.0, 1.0, 1.0]), nested([1.04, 1.03, 1.02]))
    assert result['supported'] and result['mean_difference'] == pytest.approx(-.03)
    assert not forward(nested([1., 1., 1.]), nested([1.01, 1.01, 1.01]))['supported']
    assert not forward(nested([1., 1.01, 1.01]), nested([1.2, 1., 1.]))['supported']


def test_shaping_requires_auc_ten_percent_and_every_final_within_point_zero_two():
    from nbv.hypotheses import shaping
    curves = {'linear': [(1, 1.), (2, 1.)], 'none': [(1, 1.2), (2, 1.2)]}
    assert shaping(curves, {'linear': nested([1., 1., 1.]), 'none': nested([1.02]*3)})['supported']
    assert not shaping(curves, {'linear': nested([1.]*3), 'none': nested([1.03]*3)})['supported']
    assert not shaping({'linear': curves['linear'], 'none': [(1, 1.01), (2, 1.01)]},
                       {'linear': nested([1.]*3), 'none': nested([1.]*3)})['supported']
