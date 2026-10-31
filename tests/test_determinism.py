from pathlib import Path

import numpy as np
import pytest

from nbv import env as nbv_env
from test_env import synthetic


def random_episode(e, seed):
    obs, info = e.reset(seed=seed)
    rng = np.random.default_rng(seed)
    trace, done = [info["variant"]], False
    while not done:
        a = int(rng.choice(np.flatnonzero(e.action_masks())))
        obs, r, term, trunc, info = e.step(a)
        trace.append((a, round(r, 9), round(info["cycle_time"], 9)))
        done = term or trunc
    return trace


def test_same_seed_same_random_episode_synthetic():
    data = {v: synthetic(v, seed=i) for i, v in enumerate(("1_v0", "2_v0", "3_v0"))}
    make = lambda: nbv_env.NBVEnv(list(data), loader=data.__getitem__)  # noqa: E731
    assert random_episode(make(), 5) == random_episode(make(), 5)
    assert random_episode(make(), 5) != random_episode(make(), 6)


@pytest.mark.skipif(not Path("data/cache/visibility/nominal").exists(), reason="needs precomputed caches")
def test_same_seed_same_episode_on_cached_variant():
    make = lambda: nbv_env.NBVEnv(["46262_v0"])  # noqa: E731  (train part)
    assert random_episode(make(), 0) == random_episode(make(), 0)
