"""Baselines: random, fixed pattern, greedy-IG variants, greedy-oracle, set-cover ILP + TSP.

Open-loop plans are evaluated with plan repair (D-032). Further baselines: Plan 001, Slice 5.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from nbv import features
from nbv.env import NominalModel


def greedy_cost_action(obs: dict, nominal: NominalModel, mask: np.ndarray, v: float, a: float) -> int | None:
    """Greedy-IG/cost on the nominal model: max predicted new samples per second (move estimate + scan).

    Returns None when no unmasked pass is predicted to add a sample.
    """
    gain = features.nominal_gain(obs, nominal).astype(float)
    gain[~mask] = 0.0
    if gain.max() <= 0:
        return None
    t = features.move_time_estimate(obs, nominal, v, a) + nominal.scan_time
    score = np.where(mask & (gain > 0), gain / t, -np.inf)
    return int(np.argmax(score))


def repair_policy(v: float, a: float) -> Callable[[dict, NominalModel, np.ndarray], int | None]:
    """Plan repair for open-loop methods (D-032): cost-aware greedy with nominal visibility."""
    return lambda obs, nominal, mask: greedy_cost_action(obs, nominal, mask, v, a)
