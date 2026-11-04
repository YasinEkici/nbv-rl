"""RL agents: Phase 1 (NumPy, no neural networks; Plan 002) and Phase 2 (Plan 003).

Agents see only feature matrices, action masks and rewards, never the environment. Agent code lives
in this package so that tests/test_privileged.py can scan it for access to privileged data.

Seeds (Plan 002, section 2.4): tuning seeds are disjoint from the official seeds of PROTOCOL v1. An
agent's random stream and the environment's seed sequence are separate; the environment sequence
depends only on the seed, so every method trains on the same variants in the same order.
"""

import numpy as np

OFFICIAL_SEEDS = tuple(range(10))
TUNING_SEEDS = (1000, 1001, 1002)


def agent_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng([seed, 1])


def env_seeds(seed: int, n: int) -> np.ndarray:
    """First `n` episode seeds of the environment stream of `seed`, the same for every method."""
    return np.random.default_rng([seed, 0]).integers(0, 2**31, size=n)
