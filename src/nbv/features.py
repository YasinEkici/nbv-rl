"""Feature functions for agents and planners: inputs are the observation and the nominal model only.

Phase 1 features (Plan 002) and Phase 2 inputs (Plan 003) are built here. No function in this
module may take the environment or the real variant; tests/test_env.py checks that the outputs do
not change when only the privileged data does.
"""

from __future__ import annotations

import numpy as np

from nbv.env import NominalModel
from nbv.robot import movej_time, wrap_pi


def nominal_gain(obs: dict, nominal: NominalModel) -> np.ndarray:
    """(P,) predicted newly measured samples per pass: nominal visibility minus what is already measured."""
    unmeasured = nominal.union & ~obs["covered"].astype(bool)
    return (nominal.visibility & unmeasured).sum(1)


def move_time_estimate(obs: dict, nominal: NominalModel, v: float, a: float) -> np.ndarray:
    """(P,) movej time from the current joint state to each pass start, joints wrapped to [-pi, pi].

    A lower bound of the environment's transfer time (it ignores the joint-limit check); NaN for
    unreachable passes.
    """
    start = nominal.q_plan[:, 0]
    return movej_time(wrap_pi(start - obs["q"][None, :]), v, a)


def measured_points(obs: dict, nominal: NominalModel) -> tuple[np.ndarray, np.ndarray]:
    """Nominal positions and normals of the samples measured so far (input for voxel features)."""
    m = obs["covered"].astype(bool)
    return nominal.points[m], nominal.normals[m]
