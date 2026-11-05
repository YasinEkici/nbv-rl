"""Potential-based reward shaping as an agent-side wrapper (D-038).

The environment's reward is the objective only (-cycle time + B * success). An agent may add
F = discount * Phi(s') - Phi(s) with Phi = 0 at terminal states; for any Phi the optimal policy is
unchanged (Ng, Harada & Russell 1999; terminal Phi = 0 in episodic tasks, Grzes 2017). Which Phi
an agent uses is an agent setting, not part of the protocol (Plan 002 compares them).

Phi is computed from the observation and the nominal model only (never the real variant):
    none            0
    linear          alpha * C            C = measured share of the nominal reachable union
    convex          alpha * C**exponent  convex and increasing on [0, 1]
    value_estimate  -T                   T = estimated remaining time (s): greedy-IG/cost on the
                                         nominal model from the current state to the target, then home
"""

from __future__ import annotations

from collections import OrderedDict
import gymnasium as gym
import numpy as np

from nbv.env import NominalModel
from nbv.robot import movej_time, wrap_pi

PHIS = ("none", "linear", "convex", "value_estimate")


def measured_share(obs: dict, nominal: NominalModel) -> float:
    """Share of the nominal reachable union that is measured (agent-visible stand-in for coverage)."""
    u = nominal.union
    return float((obs["covered"].astype(bool) & u).sum() / max(u.sum(), 1))


def _classes(nominal: NominalModel) -> tuple[np.ndarray, np.ndarray]:
    """Nominal union samples grouped by identical visibility columns: (P, classes) bool, class per sample."""
    uniq, inv = np.unique(nominal.visibility[:, nominal.union].T, axis=0, return_inverse=True)
    return uniq.T, inv.ravel()


def remaining_time(obs: dict, nominal: NominalModel, q_home: np.ndarray, v: float, a: float, target: float,
                   k_left: int, classes: tuple[np.ndarray, np.ndarray] | None = None) -> float:
    """Estimated time (s) to reach `target` of the nominal union and return home, by greedy-IG/cost.

    Moves are [-pi, pi] movej estimates (first from the current joint state, then ptp_pi); the
    rollout stops at the target, after `k_left` passes, or when no pass adds a sample.
    """
    vis, cls = classes if classes is not None else _classes(nominal)
    left = np.bincount(cls[~obs["covered"].astype(bool)[nominal.union]], minlength=vis.shape[1])
    need = left.sum() - (1 - target) * nominal.union.sum()  # samples still to measure
    avail = nominal.reachable & ~obs["used"].astype(bool)
    q, last, t = obs["q"], None, 0.0
    for _ in range(k_left):
        if need <= 1e-9:
            break
        gain = (vis * left).sum(1).astype(float)
        gain[~avail] = 0.0
        if gain.max() <= 0:
            break
        move = movej_time(wrap_pi(nominal.q_plan[:, 0] - q[None, :]), v, a) if last is None else nominal.ptp_pi[last]
        score = np.where(avail & (gain > 0), gain / (move + nominal.scan_time), -np.inf)
        j = int(np.argmax(score))
        t += float(move[j] + nominal.scan_time[j])
        need -= gain[j]
        left[vis[j]] = 0
        avail[j] = False
        last = j
    home = float(movej_time(wrap_pi(q_home - q), v, a)) if last is None else float(nominal.to_home_pi[last])
    return t + home


class PotentialShaping(gym.Wrapper):
    """Adds F = discount * Phi(s') - Phi(s) to the reward; Phi(s') = 0 when the episode terminates.

    With discount 1 the shaping terms of an episode sum to -Phi(s_0) (0 for linear and convex, since
    nothing is measured at reset). The term is reported in info["agent"]["shaping"].
    """

    def __init__(self, env: gym.Env, phi: str = "linear", alpha: float = 50.0, discount: float = 1.0,
                  exponent: float = 2.0, cache_value: bool = True, cache_capacity: int = 100000):
        super().__init__(env)
        if phi not in PHIS:
            raise ValueError(f"phi must be one of {PHIS}")
        self.phi, self.alpha, self.discount, self.exponent = phi, alpha, discount, exponent
        self._classes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._phi = 0.0
        self.cache_value, self.cache_capacity = cache_value, cache_capacity
        self._values: OrderedDict[tuple, float] = OrderedDict()
        self.cache_hits = self.cache_misses = 0
        self._variant, self._last_pass, self._observation = None, None, None

    def potential(self, obs: dict) -> float:
        base = self.env.unwrapped
        nominal = base.nominal
        if self.phi == "none":
            return 0.0
        if self.phi == "linear":
            return self.alpha * measured_share(obs, nominal)
        if self.phi == "convex":
            return self.alpha * measured_share(obs, nominal) ** self.exponent
        # Only wrapper-produced observations have the variant/used/last-pass correspondence.
        # External hypothetical observations remain supported, but bypass this cache.
        value_key = None
        if self.cache_value and self.cache_capacity > 0 and obs is self._observation:
            bits = int.from_bytes(np.packbits(obs["used"].astype(bool)).tobytes(), "little")
            value_key = (self._variant, bits, self._last_pass)
            if value_key in self._values:
                self.cache_hits += 1
                self._values.move_to_end(value_key)
                return self._values[value_key]
        key = id(nominal)
        if key not in self._classes:
            self._classes[key] = _classes(nominal)
        k_left = base.k_max - int(obs["step"][0])
        value = -remaining_time(obs, nominal, base.q_home, base.v, base.a, base.target, k_left, self._classes[key])
        self.cache_misses += 1
        if value_key is not None:
            self._values[value_key] = value
            if len(self._values) > self.cache_capacity:
                self._values.popitem(last=False)
        return value

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._variant, self._last_pass, self._observation = info["agent"]["variant"], None, obs
        self._phi = self.potential(obs)
        info["agent"]["potential"] = self._phi
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_pass, self._observation = info["agent"]["pass"], obs
        phi_next = 0.0 if terminated else self.potential(obs)
        f = self.discount * phi_next - self._phi
        self._phi = phi_next
        info["agent"]["shaping"] = f
        info["agent"]["potential"] = phi_next
        return obs, reward + f, terminated, truncated, info
