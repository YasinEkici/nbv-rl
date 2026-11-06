"""Pre-official checks on identical transitions and explicit train-only samples."""

import numpy as np
import pytest


def test_regq_effective_alpha_changes_updates_on_identical_data():
    from nbv.experiment import make_agent
    x = np.array([[1., .5, .2]])
    agents = [make_agent('regq', 3, {'alpha': a, 'eta': .001}, 1000) for a in (.01, .1)]
    for agent in agents:
        agent.start(x, np.array([True]))
        agent.step(2., x, np.array([True]), False)
    np.testing.assert_allclose(agents[0].w, .02 * x[0])
    np.testing.assert_allclose(agents[1].w, .2 * x[0])
    assert np.linalg.norm(agents[1].w - agents[0].w) > 0


def test_training_observer_is_passive_and_copies_selected_features():
    from nbv.agents.train import train
    from nbv.agents.linear import EpsilonSchedule
    from nbv.experiment import make_agent
    from test_env import make_env
    def run(observe=None):
        agent = make_agent('regq', 11, {'alpha': .01, 'eta': .001}, 1000)
        return train(agent, make_env(), seed=1000, budget=100, eval_every=100,
                     schedule=EpsilonSchedule.constant(.05), reduced=True,
                     evaluate=lambda w: {}, observe_features=observe)
    samples = []
    def observe(x):
        samples.append(x.copy())
        x[:] = 999
    a, b = run(), run(observe)
    assert len(samples) == 100 and np.shape(samples) == (100, 11)
    np.testing.assert_array_equal(a.final_weights, b.final_weights)
    assert a.episode_seeds == b.episode_seeds


def test_lemma_diagnostic_does_not_invert_a_singular_gram_matrix():
    from nbv.diagnostics import feature_statistics, sequence_overlap
    stats = feature_statistics(np.array([[1., 2.], [2., 4.]]))
    assert stats['rank'] == 1 and stats['lemma_threshold'] is None
    assert stats['mean_squared_norm'] == pytest.approx(12.5)
    assert sequence_overlap([1, 2], [1, 3, 4]) == pytest.approx(1 / 3)
    full = feature_statistics(np.eye(2))
    assert full['lemma_threshold'] == pytest.approx(1.)
