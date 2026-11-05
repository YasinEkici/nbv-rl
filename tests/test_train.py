"""The small Slice 2 loop will be extended, rather than replaced, in Slice 3."""

import numpy as np

from nbv.agents import agent_rng
from nbv.agents.linear import EpsilonSchedule
from nbv.agents.sarsa import TrueOnlineSarsaAgent
from test_env import make_env


def test_training_budget_tail_policy_and_evaluation_do_not_change_training():
    from nbv.agents.train import train
    def run(every):
        a = TrueOnlineSarsaAgent(16, .03 / 16, .9, rng=agent_rng(1000))
        seen = []
        def evaluate(w):
            seen.append(w.copy())
            w[:] = 999  # callback cannot mutate learner weights or the averaging accumulator
            return {"normalized_cycle_time": 1.}
        result = train(a, make_env(), seed=1000, budget=100, eval_every=every,
                       schedule=EpsilonSchedule.linear(.2, .01, .5, 100), evaluate=evaluate)
        return result, seen
    a, seen = run(10)
    b, _ = run(20)
    assert a.steps == b.steps == 100 and a.tail_count == 10
    assert len(a.history) == len(seen) == 10 and len(b.history) == 5
    np.testing.assert_array_equal(a.final_weights, b.final_weights)
    np.testing.assert_array_equal(a.policy_weights, b.policy_weights)
    assert a.episode_seeds == b.episode_seeds
    assert a.history[-1]["step"] == 100


def test_training_uses_reduced_features_and_keeps_diverged_seed():
    from nbv.agents.train import train
    a = TrueOnlineSarsaAgent(11, .03 / 11, .9, rng=agent_rng(1001), weight_limit=0)
    r = train(a, make_env(), seed=1001, budget=30, eval_every=10, reduced=True,
              schedule=EpsilonSchedule.constant(.05), evaluate=lambda w: {"normalized_cycle_time": 1.})
    assert r.steps == 30 and r.diverged and r.tail_count == 0
    np.testing.assert_array_equal(r.policy_weights, r.final_weights)


def test_weight_evaluation_reads_only_requested_variants_and_normalizes():
    from nbv.evaluation import evaluate_weights
    e = make_env()
    calls = []
    original = e._loader
    def load(vid):
        calls.append(vid)
        return original(vid)
    e._cache.clear()
    e._loader = load
    rows = evaluate_weights(e, ["1_v0"], np.zeros(16), {"1_v0": 10.})
    assert calls == ["1_v0"] and len(rows) == 1
    assert rows[0]["normalized_cycle_time"] == rows[0]["cycle_time"] / 10.
    assert "censored" in rows[0]
