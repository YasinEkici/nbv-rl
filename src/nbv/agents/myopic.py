"""Myopic agent (contextual-bandit equivalent): regression on the one-step target, no bootstrap.

The objective-only reward has no coverage term, so the caller passes the shaped reward computed at
discount 1 (r + Phi(s') - Phi(s), plus B at success): a learned greedy-IG/cost (Plan 002, section 2.1).
"""

from __future__ import annotations

import numpy as np

from nbv.agents.linear import LinearAgent


class MyopicAgent(LinearAgent):
    def __init__(self, d: int, alpha: float, rng: np.random.Generator | None = None, **kw):
        super().__init__(d, alpha, 0.0, rng, **kw)

    def step(self, reward: float, phi_next: np.ndarray, mask_next: np.ndarray, done: bool) -> int | None:
        self._apply(self.w + self.alpha * (reward - float(self.w @ self._x)) * self._x)
        a_next, self._x = self._next(phi_next, mask_next, done)
        return a_next
