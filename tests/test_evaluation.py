import numpy as np

from nbv import evaluation


def steps(*rows):
    return [dict(zip(("t_move", "t_scan", "coverage"), r)) for r in rows]


def test_auc_step_function():
    st = steps((10.0, 10.0, 0.5), (20.0, 20.0, 1.0))
    # 0 for 20 s, 0.5 for 40 s, 1.0 for the remaining 120 s of the 180 s horizon.
    assert np.isclose(evaluation.auc_coverage_time(st), (0.5 * 40 + 120) / 180)
    assert evaluation.auc_coverage_time([]) == 0.0


def test_auc_clips_at_horizon():
    st = steps((100.0, 100.0, 1.0))
    assert evaluation.auc_coverage_time(st) == 0.0


def test_coverage_at_uses_final_value_after_end():
    st = steps((1, 1, 0.3), (1, 1, 0.6))
    assert evaluation.coverage_at(st, 1) == 0.3
    assert evaluation.coverage_at(st, 10) == 0.6


def test_hierarchical_bootstrap_constant_and_levels():
    const = {"a": {"v0": [2.0, 2.0], "v1": [2.0]}, "b": {"v0": [2.0]}}
    r = evaluation.hierarchical_bootstrap(const, n_boot=200)
    assert r["mean"] == r["ci_low"] == r["ci_high"] == 2.0
    data = {"a": {"v0": [0.0], "v1": [0.0]}, "b": {"v0": [10.0], "v1": [10.0]}}
    r = evaluation.hierarchical_bootstrap(data, n_boot=2000)
    assert r["mean"] == 5.0 and r["ci_low"] == 0.0 and r["ci_high"] == 10.0  # parts resampled


def test_nested_mean_weights_parts_equally():
    data = {"a": {f"v{i}": [1.0] for i in range(5)}, "b": {"v0": [3.0]}}
    assert evaluation.nested_mean(data) == 2.0


def test_paired_difference_matches_variants():
    a = {"p": {"v0": [3.0, 5.0], "v1": [1.0]}}
    b = {"p": {"v0": [1.0], "v2": [0.0]}}
    assert evaluation.paired_difference(a, b) == {"p": {"v0": [3.0]}}


def test_iqm():
    assert evaluation.iqm(np.arange(8)) == 3.5
