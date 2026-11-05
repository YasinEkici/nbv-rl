"""Linear action-value agents: Q(s, a) = w . phi(s, a), epsilon-greedy over unmasked actions.

An agent receives `phi` of shape (actions, d) and a boolean `mask`; rows of masked actions are never
read, so they may hold NaN. `start` begins an episode and `step` learns from one transition; both
return the next action, or None when the episode is over or no action is left.

A weight update that gives NaN/inf or a norm above `weight_limit` sets `diverged`, keeps the last
finite weights and stops learning (Plan 002, section 2.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WEIGHT_LIMIT = 1e4


class TailAverage:
    """Post-update mean over the last 10% of a step budget, with D-046 divergence fallback."""

    def __init__(self, budget: int):
        if budget < 1:
            raise ValueError("budget must be positive")
        self.start = budget - max(1, budget // 10)
        self.count = 0
        self.mean: np.ndarray | None = None

    def observe(self, step: int, weights: np.ndarray, *, diverged: bool) -> None:
        if step > self.start and not diverged and np.isfinite(weights).all():
            self.count += 1
            if self.mean is None:
                self.mean = weights.copy()
            else:
                self.mean += (weights - self.mean) / self.count

    def policy(self, last_finite: np.ndarray) -> np.ndarray:
        return (last_finite if self.mean is None else self.mean).copy()


@dataclass(frozen=True)
class EpsilonSchedule:
    """Exploration rate as a function of the environment step."""

    start: float
    end: float
    decay_steps: int

    @classmethod
    def constant(cls, epsilon: float) -> EpsilonSchedule:
        return cls(epsilon, epsilon, 1)

    @classmethod
    def linear(cls, start: float, end: float, frac: float, budget: int) -> EpsilonSchedule:
        """From `start` to `end` over the first `frac` of `budget` steps, then `end`."""
        return cls(start, end, int(frac * budget))

    def __call__(self, step: int) -> float:
        if step >= self.decay_steps:
            return self.end
        return float(self.start + (self.end - self.start) * step / self.decay_steps)


class LinearAgent:
    def __init__(self, d: int, alpha: float, gamma: float = 1.0, rng: np.random.Generator | None = None,
                 weight_limit: float = WEIGHT_LIMIT):
        self.d, self.alpha, self.gamma, self.weight_limit = d, alpha, gamma, weight_limit
        self.rng = rng if rng is not None else np.random.default_rng()
        self.w = np.zeros(d)
        self.epsilon = 0.0
        self.diverged = False
        self._x = np.zeros(d)

    # --- action selection --------------------------------------------------------------------

    def greedy(self, phi: np.ndarray, mask: np.ndarray) -> int | None:
        """Best unmasked action, lowest index on ties; uses no randomness (evaluation with epsilon = 0)."""
        idx = np.flatnonzero(mask)
        return int(idx[np.argmax(phi[idx] @ self.w)]) if len(idx) else None

    def select(self, phi: np.ndarray, mask: np.ndarray) -> int | None:
        """Epsilon-greedy over unmasked actions, random on ties."""
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return None
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(idx))
        q = phi[idx] @ self.w
        return int(self.rng.choice(idx[q == q.max()]))

    # --- learning ----------------------------------------------------------------------------

    def _reset_episode(self) -> None:
        """Episode state of a subclass (traces)."""

    def start(self, phi: np.ndarray, mask: np.ndarray) -> int | None:
        self._reset_episode()
        a = self.select(phi, mask)
        self._x = phi[a].copy() if a is not None else np.zeros(self.d)
        return a

    def step(self, reward: float, phi_next: np.ndarray, mask_next: np.ndarray, done: bool) -> int | None:
        raise NotImplementedError

    def _next(self, phi_next: np.ndarray, mask_next: np.ndarray, done: bool) -> tuple[int | None, np.ndarray]:
        """Next action and its features; zero features after the end of the episode."""
        a = None if done else self.select(phi_next, mask_next)
        return a, (phi_next[a].copy() if a is not None else np.zeros(self.d))

    def _apply(self, w_new: np.ndarray) -> None:
        if self.diverged:
            return
        if not np.isfinite(w_new).all() or np.linalg.norm(w_new) > self.weight_limit:
            self.diverged = True
            return
        self.w = w_new
