"""Independent feature definitions, edge cases, scaling and nominal-only access."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from nbv import features, shaping
from test_env import make_env


def reference(obs, nom, mask, k_max):
    """Sample-level definition, deliberately without class compression."""
    measured = obs["covered"].astype(bool)
    total = max(int(nom.union.sum()), 1)
    gain = (nom.visibility & nom.union & ~measured).sum(1) / total
    move = features.move_time_estimate(obs, nom, 1.05, 1.4)
    rate = np.where(mask, gain / (move + nom.scan_time), 0)
    best = max(rate[mask], default=0)
    max_gain = max(gain[mask], default=0)
    share = shaping.measured_share(obs, nom)
    out = np.zeros((len(mask), 16))
    for a in np.flatnonzero(mask):
        after = measured | nom.visibility[a]
        rates = []
        for b in np.flatnonzero(mask):
            if a != b:
                new = (nom.visibility[b] & nom.union & ~after).sum() / total
                rates.append(new / (nom.ptp_pi[a, b] + nom.scan_time[b]))
        finish = float(gain[a] >= max(0, .95 - share))
        remaining, needed = 1 - share, max(0, .95 - share) / .95
        out[a] = [1, gain[a], gain[a] / max_gain if max_gain else 0, move[a] / 10,
                  nom.scan_time[a] / 10, rate[a] / best if best else 0, finish,
                  nom.to_home_pi[a] / 10, max(rates, default=0) / best if best else 0,
                  remaining, needed, obs["step"][0] / k_max, gain[a] * needed,
                  finish * nom.to_home_pi[a] / 10, move[a] / 10 * remaining, nom.direction[a, 2]]
    return out


def test_features_match_independent_sample_level_definition():
    e = make_env()
    obs, _ = e.reset()
    f = features.ActionFeatures(e.nominal, e.k_max)
    for action in (None, 0, 1):
        if action is not None:
            obs, *_ = e.step(action)
        got = f(obs, e.action_masks())
        assert got.shape == (8, 16) and np.isfinite(got).all()
        np.testing.assert_allclose(got, reference(obs, e.nominal, e.action_masks(), e.k_max), atol=1e-12)
        assert np.all(got[:, [1, 2, 5, 6, 9, 10, 11, 12]] >= 0)
        assert np.all(got[:, [1, 2, 5, 6, 9, 10, 11, 12]] <= 1)


def test_remaining_is_exactly_the_complement_of_shaping_share():
    e = make_env()
    obs, _ = e.reset()
    f = features.ActionFeatures(e.nominal, e.k_max)
    for a in (None, 0, 1):
        if a is not None:
            obs, *_ = e.step(a)
        phi = f(obs, e.action_masks())
        np.testing.assert_allclose(1 - phi[e.action_masks(), 9], shaping.measured_share(obs, e.nominal), atol=1e-15)
        np.testing.assert_allclose(phi[e.action_masks()] @ np.array([.5] + [0.] * 8 + [-.5] + [0.] * 6),
                                   .01 * 50 * shaping.measured_share(obs, e.nominal), atol=1e-15)


def test_time_scaling_preserves_ratios_from_home_and_after_a_pass():
    e = make_env()
    obs, _ = e.reset()
    nom = e.nominal
    doubled = dataclasses.replace(nom, **{k: 2 * getattr(nom, k) for k in
                                         ("scan_time", "ptp_pi", "from_home_pi", "to_home_pi")})
    a, b = features.ActionFeatures(nom, e.k_max), features.ActionFeatures(doubled, e.k_max)
    for action in (None, 0):
        if action is not None:
            obs, *_ = e.step(action)
        x, y = a(obs, e.action_masks()), b(obs, e.action_masks())
        np.testing.assert_allclose(x[:, [2, 5, 8]], y[:, [2, 5, 8]], atol=1e-12)
        np.testing.assert_allclose(2 * x[:, [3, 4, 7]], y[:, [3, 4, 7]], atol=1e-12)


def test_no_remaining_pass_and_zero_gain_are_finite():
    e = make_env()
    obs, _ = e.reset()
    f = features.ActionFeatures(e.nominal, e.k_max)
    only = np.zeros(8, bool)
    only[0] = True
    assert f(obs, only)[0, 8] == 0
    for mask in (only, np.zeros(8, bool)):
        x = f(dict(obs, covered=np.ones_like(obs["covered"])), mask)
        assert np.isfinite(x).all() and np.all(x[:, [1, 2, 5, 8, 9, 10]] == 0)


def test_next_rate_uses_current_denominator_and_reduced_subset_is_exact():
    e = make_env()
    obs, _ = e.reset()
    full = features.ActionFeatures(e.nominal, e.k_max)
    small = features.ActionFeatures(e.nominal, e.k_max, reduced=True)
    x = full(obs, e.action_masks())
    assert np.ptp(x[e.action_masks(), 8]) > .01  # not an identically-one next-state normalization
    np.testing.assert_array_equal(small(obs, e.action_masks()), x[:, features.REDUCED_INDICES])


def test_features_do_not_take_or_read_privileged_data():
    class Poison:
        def __getattr__(self, name):
            raise AssertionError(name)
    e = make_env()
    obs, _ = e.reset()
    f = features.ActionFeatures(e.nominal, e.k_max)
    expected = f(obs, e.action_masks())
    e.data = dataclasses.replace(e.data, real=Poison())
    np.testing.assert_array_equal(f(obs, e.action_masks()), expected)


@pytest.mark.skipif(not Path("data/cache/visibility/nominal").exists(), reason="needs precomputed caches")
def test_cached_train_features_are_bounded():
    from nbv.env import NBVEnv
    e = NBVEnv(["65149_v0"])
    obs, _ = e.reset()
    f = features.ActionFeatures(e.nominal, e.k_max)
    done = False
    while not done:
        x = f(obs, e.action_masks())
        assert np.isfinite(x).all() and np.abs(x).max() < 10
        obs, _, done, _, _ = e.step(int(np.flatnonzero(e.action_masks())[0]))
