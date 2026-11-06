"""Small reusable episode/step loop for the Slice 2 feature check (extend in Slice 3).

Learning reads observations, rewards and nominal CAD only. Evaluation is an injected callback
with a copy of weights; it never shares the learner's random stream or environment. Files and
experiment orchestration are outside this loop.
"""

from collections.abc import Callable
from dataclasses import dataclass
import time

import numpy as np

from nbv.features import ActionFeatures
from nbv.agents.linear import EpsilonSchedule, LinearAgent, TailAverage


@dataclass
class TrainingResult:
    steps: int
    policy_weights: np.ndarray
    final_weights: np.ndarray
    history: list[dict]
    episode_seeds: list[int]
    tail_count: int
    diverged: bool
    wall_s: float
    learning_s: float
    evaluation_s: float


def train(agent: LinearAgent, env, *, seed: int, budget: int, eval_every: int,
          schedule: EpsilonSchedule, evaluate: Callable[[np.ndarray], dict],
          reduced: bool = False, reward_scale: float = .01, stop_on_nonfinite: bool = False,
          observe_features: Callable[[np.ndarray], None] | None = None) -> TrainingResult:
    if budget < 1 or eval_every < 1:
        raise ValueError("budget and evaluation interval must be positive")
    base = env.unwrapped
    rng = np.random.default_rng([seed, 0])
    tail = TailAverage(budget)
    history, episode_seeds, cache = [], [], {}
    step, done, started = 0, True, time.perf_counter()
    evaluation_s = 0.
    while step < budget:
        if done:
            ep_seed = int(rng.integers(0, 2**31))
            episode_seeds.append(ep_seed)
            obs, _ = env.reset(seed=ep_seed)
            key = id(base.nominal)
            if key not in cache:
                cache[key] = ActionFeatures(base.nominal, base.k_max, reduced=reduced, target=base.target)
            feature = cache[key]
            agent.epsilon = schedule(step)
            action = agent.start(feature(obs, base.action_masks()), base.action_masks())
        if observe_features is not None:
            observe_features(agent._x.copy())
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1
        agent.epsilon = schedule(step)
        mask = base.action_masks()
        phi = np.zeros((len(mask), agent.d)) if done else feature(obs, mask)
        action = agent.step(reward_scale * reward, phi, mask, done)
        if stop_on_nonfinite and agent.divergence_reason == "nonfinite":
            raise FloatingPointError("nonfinite learner update; requested stop condition")
        tail.observe(step, agent.w, diverged=agent.diverged)
        if step % eval_every == 0 or step == budget:
            weights = tail.policy(agent.w)
            evaluation_started = time.perf_counter()
            metrics = evaluate(weights)
            evaluation_s += time.perf_counter() - evaluation_started
            history.append({"step": step, "weight_norm": float(np.linalg.norm(agent.w)),
                            "diverged": agent.diverged, **metrics})
    wall_s = time.perf_counter() - started
    return TrainingResult(step, tail.policy(agent.w), agent.w.copy(), history, episode_seeds,
                          tail.count, agent.diverged, wall_s, wall_s - evaluation_s, evaluation_s)
