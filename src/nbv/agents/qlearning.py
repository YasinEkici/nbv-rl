"""Linear Q-learning with L2 regularization, adapted from RegQ (Lim & Lee, arXiv 2202.05404).

Linear function approximation + off-policy learning + bootstrapping can diverge, so the weight norm
is watched (`LinearAgent._apply`). The behaviour policy is the same epsilon-greedy as SARSA's.
"""

from __future__ import annotations

import numpy as np

from nbv.agents.linear import LinearAgent


class RegQAgent(LinearAgent):
    def __init__(self, d: int, alpha: float, eta: float, gamma: float = 1.0, rng: np.random.Generator | None = None,
                 **kw):
        super().__init__(d, alpha, gamma, rng, **kw)
        self.eta = eta

    def step(self, reward: float, phi_next: np.ndarray, mask_next: np.ndarray, done: bool) -> int | None:
        best = 0.0
        if not done and mask_next.any():
            best = float((phi_next[mask_next] @ self.w).max())
        delta = reward + self.gamma * best - float(self.w @ self._x)
        self._apply(self.w + self.alpha * (delta * self._x - self.eta * self.w))
        a_next, self._x = self._next(phi_next, mask_next, done)
        return a_next
