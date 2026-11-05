"""Feature functions for agents and planners: inputs are the observation and the nominal model only.

Phase 1 features (Plan 002) and Phase 2 inputs (Plan 003) are built here. No function in this
module may take the environment or the real variant; tests/test_env.py checks that the outputs do
not change when only the privileged data does.
"""

from __future__ import annotations

import numpy as np

from nbv.env import NominalModel
from nbv.robot import movej_time, wrap_pi
from nbv.shaping import _classes, measured_share

FEATURE_NAMES = ("bias", "gain", "gain_rel", "t_move", "t_scan", "rate_rel", "finishes", "t_home",
                 "next_rate", "remaining", "needed", "step_frac", "gain_needed", "finishes_home",
                 "move_remaining", "elevation")
REDUCED_INDICES = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11])


class ActionFeatures:
    """Plan 002's 16 features, or its predeclared 11-feature subset; masked rows are zero.

    Construct once per nominal model. Identical visibility columns are compressed once, but
    counts of unmeasured samples within each class remain exact (partial classes are allowed).
    Time features use the planner's cached seconds and the fixed 10 s scale, without fitted
    normalization. Legal observations are at home or a pass end, so the wrapped current-state
    movej estimate is exactly a from_home_pi entry or a ptp_pi row. Joint-state matching chooses
    that row even when the executed sweep was shifted by 2*pi; the multi-turn joint limit is
    deliberately ignored, as in move_time_estimate. This also preserves time-unit scaling.
    """

    def __init__(self, nominal: NominalModel, k_max: int = 24, *, reduced: bool = False, target: float = .95):
        self.nominal, self.k_max, self.reduced, self.target = nominal, k_max, reduced, target
        vis, self.classes = _classes(nominal)
        self.total = max(int(nominal.union.sum()), 1)
        # Counts are integers <= 20000 in PROTOCOL v1, exactly representable in float32.
        # Single-precision BLAS speeds up overlap counts without approximating a feature.
        self.vis = vis.astype(np.float32 if self.total < 2**24 else np.float64)
        self.ends = np.nan_to_num(nominal.q_plan[:, -1])
        self.future_time = nominal.ptp_pi + nominal.scan_time[None, :]

    def __call__(self, obs: dict, mask: np.ndarray) -> np.ndarray:
        nom = self.nominal
        idx = np.flatnonzero(mask)
        out = np.zeros((len(mask), 16))
        if not len(idx):
            return out[:, REDUCED_INDICES] if self.reduced else out
        used = obs["used"].astype(bool)
        if not used.any():
            move = nom.from_home_pi[idx]
        else:
            previous = np.flatnonzero(used & nom.reachable)
            distance = np.abs(wrap_pi(self.ends[previous] - obs["q"])).max(1)
            last = previous[np.argmin(distance)]
            if distance.min() > 1e-7:
                raise ValueError("feature observation must be at home or a nominal pass end modulo 2*pi")
            move = nom.ptp_pi[last, idx]
        left = np.bincount(self.classes[~obs["covered"].astype(bool)[nom.union]],
                           minlength=self.vis.shape[1]).astype(self.vis.dtype)
        V = self.vis[idx]
        counts = (V @ left).astype(float)
        gain = counts / self.total
        rate = gain / (move + nom.scan_time[idx])
        best_gain, best_rate = gain.max(), rate.max()
        share = measured_share(obs, nom)  # exactly the same C as linear Phi; never clipped
        remaining, needed = 1 - share, max(0., self.target - share) / self.target
        finishes = gain >= max(0., self.target - share)
        out[idx, 0] = 1
        out[idx, 1] = gain
        out[idx, 2] = gain / best_gain if best_gain > 0 else 0
        out[idx, 3] = move / 10
        out[idx, 4] = nom.scan_time[idx] / 10
        out[idx, 5] = rate / best_rate if best_rate > 0 else 0
        out[idx, 6] = finishes
        out[idx, 7] = nom.to_home_pi[idx] / 10
        out[idx, 9] = remaining
        out[idx, 10] = needed
        out[idx, 11] = obs["step"][0] / self.k_max
        if self.reduced:
            return out[:, REDUCED_INDICES]
        if best_rate > 0 and len(idx) > 1:
            # Row a, column b: samples b still adds after nominally executing a.
            overlap = (V * left) @ V.T
            future_gain = np.maximum(counts[None, :] - overlap, 0) / self.total
            future_rate = future_gain / self.future_time[np.ix_(idx, idx)]
            np.fill_diagonal(future_rate, 0)
            out[idx, 8] = future_rate.max(1) / best_rate
        out[idx, 12] = gain * needed
        out[idx, 13] = finishes * nom.to_home_pi[idx] / 10
        out[idx, 14] = move / 10 * remaining
        out[idx, 15] = nom.direction[idx, 2]
        return out


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
