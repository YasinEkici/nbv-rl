"""True online SARSA(lambda) with linear function approximation and dutch traces.

Adapted from van Seijen et al., JMLR 17(145), 2016. With a fixed action sequence the weights equal
those of the online lambda-return algorithm (tests/test_agents.py); with lambda = 0 it is one-step SARSA.
"""

from __future__ import annotations

import numpy as np

from nbv.agents.linear import LinearAgent


class TrueOnlineSarsaAgent(LinearAgent):
    def __init__(self, d: int, alpha: float, lam: float, gamma: float = 1.0, rng: np.random.Generator | None = None,
                 **kw):
        super().__init__(d, alpha, gamma, rng, **kw)
        self.lam = lam
        self._reset_episode()

    def _reset_episode(self) -> None:
        self._z = np.zeros(self.d)
        self._q_old = 0.0

    def step(self, reward: float, phi_next: np.ndarray, mask_next: np.ndarray, done: bool) -> int | None:
        a_next, x_next = self._next(phi_next, mask_next, done)
        x, g, al = self._x, self.gamma, self.alpha
        q, q_next = float(self.w @ x), float(self.w @ x_next)
        delta = reward + g * q_next - q
        self._z = g * self.lam * self._z + (1.0 - al * g * self.lam * float(self._z @ x)) * x
        self._apply(self.w + al * (delta + q - self._q_old) * self._z - al * (q - self._q_old) * x)
        self._q_old, self._x = q_next, x_next
        return a_next
