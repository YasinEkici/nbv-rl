import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from nbv import baselines
from nbv import env as nbv_env
from test_env import make_env, synthetic


def brute_force_tsp(start, C, end):
    best = (math.inf, None)
    for perm in itertools.permutations(range(len(start))):
        c = start[perm[0]] + sum(C[perm[i], perm[i + 1]] for i in range(len(perm) - 1)) + end[perm[-1]]
        best = min(best, (c, list(perm)))
    return best


def test_held_karp_matches_brute_force():
    rng = np.random.default_rng(0)
    for n in (2, 4, 6):
        s, e, C = rng.uniform(1, 5, n), rng.uniform(1, 5, n), rng.uniform(1, 5, (n, n))
        order, cost = baselines.held_karp(s, C, e)
        bf_cost, _ = brute_force_tsp(s, C, e)
        assert sorted(order) == list(range(n))
        assert np.isclose(cost, bf_cost)


def test_routing_tsp_returns_a_tour_close_to_optimal():
    rng = np.random.default_rng(1)
    n = 7
    s, e, C = rng.uniform(1, 5, n), rng.uniform(1, 5, n), rng.uniform(1, 5, (n, n))
    order, cost = baselines.routing_tsp(s, C, e, time_limit_s=1.0)
    assert sorted(order) == list(range(n))
    assert cost <= brute_force_tsp(s, C, e)[0] * 1.05 + 1e-3


def test_set_cover_is_minimal_and_feasible():
    d = synthetic()
    nom = d.nominal
    rows = np.flatnonzero(nom.reachable)
    for frac in (0.95, 1.0):
        res = baselines.set_cover(nom.visibility, nom.union, rows, frac, time_limit_s=10)
        cov = nom.visibility[res.passes].any(0) & nom.union
        assert cov.sum() >= math.ceil(frac * nom.union.sum() - 1e-9)
        assert res.status == "OPTIMAL"
        # No smaller subset reaches the target.
        smaller = len(res.passes) - 1
        for sub in itertools.combinations(rows, smaller):
            assert (nom.visibility[list(sub)].any(0) & nom.union).sum() < math.ceil(frac * nom.union.sum() - 1e-9)


def test_fixed_pattern_is_reachable_and_distinct():
    nom = synthetic().nominal
    nom = nbv_env.NominalModel(**{**nom.__dict__, "direction": np.vstack([[0, 0, 1.0], np.eye(3), -np.eye(3)[:2], [0.6, 0.6, 0.5], [0.0, 0.7, 0.7]])})
    seq = baselines.fixed_pattern(nom)
    assert len(seq) == len(set(seq)) and all(nom.reachable[seq])
    assert seq[0] == 0  # nearest to the zenith


def _sequences(e, vid):
    """Pass sequences of the deterministic methods on one variant (truncated runs)."""
    out = {}
    for name, make in {
        "greedy_ig": lambda env: baselines.greedy_ig_action,
        "greedy_cost": lambda env: baselines.greedy_cost_policy(env.v, env.a),
        "greedy_oracle": lambda env: baselines.greedy_oracle_policy(env.data.real),
    }.items():
        out[name] = [s["pass"] for s in nbv_env.evaluate_policy(e, vid, make)["steps"]]
    e.reset(options={"variant": vid})
    plan, _ = baselines.ilp_tsp_plan(e.nominal, time_limit_s=10)
    out["ilp_tsp"] = [s["pass"] for s in nbv_env.evaluate_sequence(e, vid, plan, baselines.repair_policy(e.v, e.a))["steps"]]
    out["fixed"] = [s["pass"] for s in nbv_env.evaluate_sequence(e, vid, baselines.fixed_pattern(e.nominal), baselines.repair_policy(e.v, e.a))["steps"]]
    return out


def _check_upper_bound(e, vid, k_max=6):
    e.reset(options={"variant": vid})
    real, rows = e.data.real, np.flatnonzero(e.nominal.reachable)
    seqs = _sequences(e, vid)
    for k in range(1, k_max + 1):
        ub = baselines.max_coverage(real.visibility, real.union, rows, k, time_limit_s=30)
        for name, seq in seqs.items():
            cov = (real.visibility[seq[:k]].any(0) & real.union).sum() / real.union.sum() if seq else 0.0
            assert cov <= ub.value + 1e-9, (name, k, cov, ub.value)
    first = lambda s: (real.visibility[s[0]] & real.union).sum()  # noqa: E731
    assert first(seqs["greedy_oracle"]) >= first(seqs["greedy_ig"])  # guaranteed only at k = 1


def test_max_coverage_ilp_bounds_every_method_synthetic():
    _check_upper_bound(make_env(), "1_v0", k_max=4)


@pytest.mark.skipif(not Path("data/cache/visibility/nominal").exists(), reason="needs precomputed caches")
def test_max_coverage_ilp_bounds_every_method_on_cached_variants():
    for vid in ("46262_v0", "591247_v1"):  # train, val
        _check_upper_bound(nbv_env.NBVEnv([vid]), vid)


def test_mcts_returns_valid_action_and_is_seeded():
    def run(seed):
        e = make_env()
        planner = baselines.MCTSPlanner(e.v, e.a, np.random.default_rng(seed), e.k_max, e.target, n_sim=40)
        return [s["pass"] for s in nbv_env.evaluate_policy(e, e.variants[0], lambda env: planner)["steps"]]

    a = run(0)
    assert a == run(0)
    assert len(a) == len(set(a)) and all(synthetic().nominal.reachable[a])


def test_random_tsp_plan_uses_k_distinct_reachable_passes():
    nom = synthetic().nominal
    plan, info = baselines.random_tsp_plan(nom, 4, np.random.default_rng(0))
    assert len(plan) == 4 == len(set(plan)) and all(nom.reachable[plan])
    assert info["tsp"] == "held_karp"


def test_joint_opt_is_optimal_on_small_instance_and_never_worse_than_ilp_tsp():
    nom = synthetic().nominal
    rows = np.flatnonzero(nom.reachable)
    need = math.ceil(0.95 * nom.union.sum() - 1e-9)
    best = math.inf
    for r in range(1, len(rows) + 1):
        for sub in itertools.combinations(rows, r):
            if (nom.visibility[list(sub)].any(0) & nom.union).sum() < need:
                continue
            for perm in itertools.permutations(sub):
                best = min(best, baselines.planner_cost(nom, list(perm)))
    ilp, _ = baselines.ilp_tsp_plan(nom, time_limit_s=10)
    plan, info = baselines.joint_opt_plan(nom, time_limit_s=30, hint=ilp, workers=1)
    assert info["joint_status"] == "OPTIMAL"
    assert (nom.visibility[plan].any(0) & nom.union).sum() >= need
    assert np.isclose(baselines.planner_cost(nom, plan), best, atol=5e-3)  # ms rounding
    assert baselines.planner_cost(nom, plan) <= baselines.planner_cost(nom, ilp) + 5e-3
    assert info["joint_bound_s"] <= info["joint_obj_s"] + 1e-9


@pytest.mark.skipif(not Path("data/cache/visibility/nominal").exists(), reason="needs precomputed caches")
def test_ilp_tsp_plan_is_reproducible():
    e = nbv_env.NBVEnv(["65149_v0"])
    e.reset(options={"variant": "65149_v0"})
    plans = {tuple(baselines.ilp_tsp_plan(e.nominal)[0]) for _ in range(3)}
    assert len(plans) == 1
