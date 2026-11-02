import dataclasses

import numpy as np
import pytest

from nbv import shaping
from test_env import make_env, synthetic


def run(env, seed, k_max=None):
    """Random episode; returns per-step (env reward, shaped reward, info) and the reset info."""
    base = env.unwrapped
    if k_max is not None:
        base.k_max = k_max
    obs, info0 = env.reset(seed=seed, options={"variant": base.variants[0]})
    rng, out, done = np.random.default_rng(seed), [], False
    while not done:
        obs, r, term, trunc, info = env.step(int(rng.choice(np.flatnonzero(base.action_masks()))))
        env_r = sum(info["agent"]["reward_parts"].values())
        out.append((env_r, r, info))
        done = term or trunc
    return out, info0


@pytest.mark.parametrize("phi", shaping.PHIS)
def test_shaped_return_differs_from_env_return_by_minus_phi0(phi):
    for seed in range(4):
        for k_max in (None, 2):  # success and end at K_max
            w = shaping.PotentialShaping(make_env(), phi=phi)
            steps, info0 = run(w, seed, k_max)
            env_ret, shaped_ret = sum(s[0] for s in steps), sum(s[1] for s in steps)
            phi0 = info0["agent"]["potential"]
            assert np.isclose(shaped_ret - env_ret, -phi0)
            assert steps[-1][2]["agent"]["potential"] == 0.0  # terminal potential
            assert np.isclose(env_ret, -w.unwrapped.cycle_time + w.unwrapped.bonus * steps[-1][2]["agent"]["success"])
            if phi in ("none", "linear", "convex"):
                assert phi0 == 0.0  # nothing measured at reset


def test_value_estimate_potential_is_minus_a_positive_time():
    w = shaping.PotentialShaping(make_env(), phi="value_estimate")
    _, info0 = w.reset(options={"variant": "1_v0"})
    assert info0["agent"]["potential"] < 0


def test_discounted_shaping_term():
    w = shaping.PotentialShaping(make_env(), phi="linear", discount=0.9)
    steps, info0 = run(w, 0)
    phis = [info0["agent"]["potential"]] + [s[2]["agent"]["potential"] for s in steps]
    for k, (_, _, info) in enumerate(steps):
        assert np.isclose(info["agent"]["shaping"], 0.9 * phis[k + 1] - phis[k])


def test_convex_potential_is_convex_and_increasing_in_coverage():
    w = shaping.PotentialShaping(make_env(), phi="convex")
    obs, _ = w.reset(options={"variant": "1_v0"})
    nom = w.unwrapped.nominal
    idx = np.flatnonzero(nom.union)
    vals = []
    for m in np.linspace(0, len(idx), 6).astype(int):
        o = dict(obs, covered=np.zeros_like(obs["covered"]))
        o["covered"][idx[:m]] = 1
        vals.append(w.potential(o))
    d = np.diff(vals)
    assert np.all(d >= 0) and np.all(np.diff(d) >= -1e-9)


class _Poison:
    def __getattr__(self, name):
        raise AssertionError(f"privileged data accessed: real.{name}")


@pytest.mark.parametrize("phi", shaping.PHIS)
def test_potential_uses_only_agent_visible_data(phi):
    a, b = synthetic(real_seed=11), synthetic(real_seed=99)
    assert not np.array_equal(a.real.visibility, b.real.visibility)
    wa, wb = shaping.PotentialShaping(make_env(a), phi=phi), shaping.PotentialShaping(make_env(b), phi=phi)
    obs, _ = wa.reset(options={"variant": "1_v0"})
    wb.reset(options={"variant": "1_v0"})
    for w in (wa, wb):  # the real part is unreadable from here on
        w.unwrapped.data = dataclasses.replace(w.unwrapped.data, real=_Poison())
    obs = dict(obs, covered=(np.arange(len(obs["covered"])) % 3 == 0).astype(np.int8))
    assert wa.potential(obs) == wb.potential(obs)
