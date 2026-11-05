"""Experiment settings, fixed selection rules and eligible-input provenance."""

import json

import numpy as np
import pytest


def test_grids_match_the_predeclared_twelve_settings_per_method():
    from nbv.experiment import grid_settings, make_agent
    for method in ("sarsa", "regq", "myopic"):
        settings = grid_settings(method)
        assert len(settings) == len({json.dumps(s, sort_keys=True) for s in settings}) == 12
        assert {s["alpha"] for s in settings} == {.01, .03, .1}
        assert {s["epsilon"] for s in settings} == {"linear", "constant"}
    assert {s["eta"] for s in grid_settings("regq")} == {.001, .01}
    assert {s["phi_alpha"] for s in grid_settings("myopic")} == {25., 50.}
    for method in ("sarsa", "regq", "myopic"):
        a = make_agent(method, 11, grid_settings(method)[0], 1000)
        assert a.alpha == (.01 / 11 if method == "sarsa" else .01)


def test_selection_applies_score_band_then_seed_spread_then_step_size():
    from nbv.experiment import select_setting
    points = [dict(setting={"alpha": .1}, scores=[1.1, 1.1, 1.1]),
              dict(setting={"alpha": .01}, scores=[1.104, 1.104, 1.104]),
              dict(setting={"alpha": .03}, scores=[1.08, 1.10, 1.12])]
    assert select_setting(points)["setting"]["alpha"] == .01
    points.append(dict(setting={"alpha": .1}, scores=[1.09, 1.09, 1.09]))
    assert select_setting(points)["setting"]["alpha"] == .1


def test_cache_manifest_reads_only_explicit_eligible_files(tmp_path):
    from nbv.evaluation import selected_cache_manifest
    selected, excluded = tmp_path / "selected.npz", tmp_path / "excluded.npz"
    np.savez_compressed(selected, a=np.arange(5))
    excluded.write_bytes(b"must not be opened as a cache")
    first = selected_cache_manifest([str(selected)])
    np.savez_compressed(selected, a=np.arange(5) + 1)
    assert first != selected_cache_manifest([str(selected)])
    assert str(excluded) not in str(first)


@pytest.mark.parametrize("method", ["sarsa", "regq", "myopic"])
def test_all_learners_use_the_same_loop_and_masked_nominal_features(method):
    from nbv.experiment import grid_settings, make_agent
    from nbv.agents.train import train
    from nbv.agents.linear import EpsilonSchedule
    from test_env import make_env
    agent = make_agent(method, 11, grid_settings(method)[0], 1000)
    result = train(agent, make_env(), seed=1000, budget=40, eval_every=2, reduced=True,
                   schedule=EpsilonSchedule.constant(.05), evaluate=lambda w: {"normalized_cycle_time": 1.})
    assert len(result.history) == 20 and np.isfinite(result.policy_weights).all()


def test_smoke_estimate_includes_all_seventy_seeds():
    from nbv.training_report import official_hours
    measurements = {(m, p, d): {"learning_s": 10., "evaluation_s": 2.}
                    for m in ("sarsa", "regq", "myopic")
                    for p in ("none", "linear", "convex", "value_estimate") for d in (11, 16)}
    assert official_hours(measurements) == pytest.approx(70 * (2000 + 2) / 20 / 3600)
