"""VAL reports retain equal part weights and show seed ranges without significance claims."""
import pytest


def test_official_summary_includes_censored_episodes_and_equal_part_weights():
    from nbv.official_report import summarize_episodes, curve_summary
    records = [{'part': 1, 'variant': '1_v0', 'normalized_cycle_time': 1., 'cycle_time': 10., 'success': True},
               {'part': 2, 'variant': '2_v0', 'normalized_cycle_time': 2., 'cycle_time': 20., 'success': False},
               {'part': 2, 'variant': '2_v1', 'normalized_cycle_time': 2., 'cycle_time': 20., 'success': False}]
    s = summarize_episodes(records)
    assert s['mean'] == pytest.approx(1.5) and s['success'] == pytest.approx(.5)
    assert s['parts'] == {'1': 1., '2': 2.}
    c = curve_summary([[{'step': 100, 'normalized_cycle_time': 1.}],
                       [{'step': 100, 'normalized_cycle_time': 3.}]])
    assert c == {'steps': [100], 'mean': [2.], 'low': [1.], 'high': [3.]}
