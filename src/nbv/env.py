"""Gymnasium NBV environment: coverage, reward, action masking (D-029, D-030).

Information model: the agent knows the nominal CAD (`NominalModel`: samples, nominal visibility,
reachability, joint plans and planner time matrices) and what has been measured so far. The real
variant's visibility and reachable union (`RealPart`) are privileged: they drive the simulation and
the coverage metrics but never enter observations.

Episode: the arm starts at q_home; each step executes one reachable, unused scan pass. The episode
terminates when primary coverage of the real reachable union reaches the target, and is truncated
after K_max passes or when no pass is left; in both cases the return home is added to the cycle
time. The step count is part of the state, so truncation at K_max is an end of the task, not a time
limit of the simulator.

Reward per step: alpha * (discount * C' - C) - beta * t_move - gamma_t * t_scan, plus the success
bonus once; at the episode end the return home is charged and, with terminal_potential_zero, the
potential is reset (- alpha * discount * C_end), so the shaping terms sum to zero over an episode.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np

from nbv import parts, robot, visibility

ENV_VERSION = "v1"


@dataclass(frozen=True)
class NominalModel:
    """Agent-visible knowledge, derived from the nominal CAD and the cell (D-029)."""

    points: np.ndarray        # (N, 3) mm, nominal sample positions
    normals: np.ndarray       # (N, 3)
    visibility: np.ndarray    # (P, N) bool, measured on the nominal mesh
    reachable: np.ndarray     # (P,) bool
    union: np.ndarray         # (N,) bool, nominal reachable union
    q_plan: np.ndarray        # (P, k, 6) rad, joint plan per pass (NaN if unreachable)
    direction: np.ndarray     # (P, 3) unit, center -> sensor
    scan_time: np.ndarray     # (P,) s
    ptp_pi: np.ndarray        # (P, P) s, planner approximation (D-024)
    from_home_pi: np.ndarray  # (P,) s
    to_home_pi: np.ndarray    # (P,) s


@dataclass(frozen=True)
class RealPart:
    """Privileged: the variant as it is. Never part of observations or features."""

    visibility: np.ndarray    # (P, N) bool, measured on the variant with the nominal passes
    union: np.ndarray         # (N,) bool, real reachable union (primary reference, D-023)
    dense_union: np.ndarray   # (N,) bool, real reachable union of the dense set (NSC reference, D-028)


@dataclass(frozen=True)
class EpisodeData:
    variant_id: str
    n_candidates: int
    nominal: NominalModel
    real: RealPart


def _nominal_points(fid: int, vis_dir: Path, parts_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Nominal sample positions and normals, cached next to the samples file."""
    path = vis_dir / "nominal" / f"points_{fid}.npz"
    if not path.exists():
        g = parts_cfg["geometry"]
        mesh = parts.load_base(Path(parts_cfg["paths"]["raw"]) / f"{fid}.ply", g["longest_mm"], g["max_edge_mm"])
        s = np.load(vis_dir / f"samples_{fid}.npz")
        p, n = visibility.SurfaceSamples(s["face"], s["bary"]).on(mesh)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, points=p, normals=n)
        os.replace(tmp, path)  # atomic: parallel workers never read a half-written file
    z = np.load(path)
    return z["points"], z["normals"]


def load_episode_data(variant_id: str, n: int, env_cfg: dict, parts_cfg: dict, dense_n: int) -> EpisodeData:
    fid = int(variant_id.split("_")[0])
    rdir, vdir = Path(env_cfg["paths"]["robot"]), Path(env_cfg["paths"]["visibility"])
    z = np.load(rdir / f"{fid}_n{n}.npz")
    reach = z["reachable"]
    nom_vis = visibility.Visibility.load(vdir / "nominal" / f"{fid}_n{n}.npz").measured()
    real_vis = visibility.Visibility.load(vdir / f"{variant_id}_n{n}.npz").measured()
    dense_vis = visibility.Visibility.load(vdir / f"{variant_id}_n{dense_n}.npz").measured()
    dense_reach = np.load(rdir / f"{fid}_n{dense_n}.npz")["reachable"]
    points, normals = _nominal_points(fid, vdir, parts_cfg)
    nominal = NominalModel(
        points=points, normals=normals, visibility=nom_vis, reachable=reach,
        union=visibility.union(nom_vis, reach), q_plan=z["q"], direction=z["direction"],
        scan_time=z["scan_time"], ptp_pi=z["ptp_pi"], from_home_pi=z["from_home_pi"], to_home_pi=z["to_home_pi"],
    )
    real = RealPart(real_vis, visibility.union(real_vis, reach), visibility.union(dense_vis, dense_reach))
    return EpisodeData(variant_id, n, nominal, real)


class NBVEnv(gym.Env):
    """Scan-pass selection on one variant per episode (`reset(options={"variant": ...})`)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        variants: list[str],
        n_candidates: int = 48,
        env_config: str = "configs/env_v1.yaml",
        robot_config: str = "configs/robot_v1.yaml",
        parts_config: str = "configs/parts_v2.yaml",
        candidates_config: str = "configs/candidates_v1.yaml",
        loader: Callable[[str], EpisodeData] | None = None,
    ):
        self.cfg = parts.load_yaml(env_config)
        self.robot_cfg = parts.load_yaml(robot_config)
        self.parts_cfg = parts.load_yaml(parts_config)
        self.dense_n = parts.load_yaml(candidates_config)["n_candidates_sensitivity"]
        self.variants = list(variants)
        self.n_candidates = n_candidates
        self.k_max = int(self.cfg["k_max"][n_candidates])
        self.pose_id = int(self.cfg["pose_id"])
        rw = self.cfg["reward"]
        self.alpha, self.beta, self.gamma_t = rw["alpha_s"], rw["beta_move"], rw["gamma_scan"]
        self.bonus, self.discount, self.scale = rw["success_bonus_s"], rw["discount"], rw["reward_scale"]
        self.phi_zero = bool(rw["terminal_potential_zero"])
        self.target = float(self.cfg["coverage_target"])
        m = self.robot_cfg["motion"]
        self.v, self.a = m["v_rad_s"], m["a_rad_s2"]
        lim = np.radians(self.robot_cfg["joint_limit_deg"])
        self.qlim = np.array([[-lim] * 6, [lim] * 6])
        self.q_home = robot.q_home(self.robot_cfg)
        self._cache: dict[str, EpisodeData] = {}
        self._loader = loader or (
            lambda vid: load_episode_data(vid, self.n_candidates, self.cfg, self.parts_cfg, self.dense_n)
        )
        first = self._data(self.variants[0])
        P, N = first.nominal.visibility.shape
        self.action_space = gym.spaces.Discrete(P)
        self.observation_space = gym.spaces.Dict({
            "covered": gym.spaces.MultiBinary(N),
            "used": gym.spaces.MultiBinary(P),
            "q": gym.spaces.Box(-lim, lim, shape=(6,), dtype=np.float64),
            "step": gym.spaces.Box(0, self.k_max, shape=(1,), dtype=np.float64),
            "nominal_coverage": gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
        })
        self._done = True

    # --- data ----------------------------------------------------------------

    def _data(self, variant_id: str) -> EpisodeData:
        if variant_id not in self._cache:
            self._cache[variant_id] = self._loader(variant_id)
        return self._cache[variant_id]

    @property
    def nominal(self) -> NominalModel:
        """Agent-visible model of the current episode's part."""
        return self.data.nominal

    # --- coverage ------------------------------------------------------------

    @staticmethod
    def _cov(covered: np.ndarray, ref: np.ndarray) -> float:
        return visibility.coverage(covered, ref)

    def coverages(self) -> dict:
        r, n = self.data.real, self.data.nominal
        return {
            "coverage": self._cov(self._covered, r.union),             # primary (D-028)
            "coverage_nsc": self._cov(self._covered, r.dense_union),
            "coverage_raw": float(self._covered.mean()),
            "nominal_coverage": self._cov(self._covered_nom, n.union),  # predicted from the CAD
        }

    # --- gym API -------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        return self.data.nominal.reachable & ~self._used

    def _obs(self) -> dict:
        return {
            "covered": self._covered.astype(np.int8),
            "used": self._used.astype(np.int8),
            "q": self._q.copy(),
            "step": np.array([self._step], dtype=np.float64),
            "nominal_coverage": np.array([self._c_nom], dtype=np.float64),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if options.get("pose_id", self.pose_id) != 0:
            raise NotImplementedError("only pose_id 0 is precomputed (Plan 003)")
        vid = options.get("variant") or self.variants[int(self.np_random.integers(len(self.variants)))]
        self.data = self._data(vid)
        P, N = self.data.nominal.visibility.shape
        self._covered, self._covered_nom = np.zeros(N, bool), np.zeros(N, bool)
        self._used = np.zeros(P, bool)
        self._q = self.q_home.copy()
        self._step, self._c, self._c_nom = 0, 0.0, 0.0
        self._t_move = self._t_scan = self._t_home = 0.0
        self._joint_motion = 0.0
        self._done = False
        return self._obs(), {"variant": vid, "action_mask": self.action_masks(), **self.coverages()}

    def step(self, action: int):
        if self._done:
            raise RuntimeError("episode is over; call reset()")
        a = int(action)
        if not self.action_masks()[a]:
            raise ValueError(f"pass {a} is masked (unreachable or already used)")
        nom, real = self.data.nominal, self.data.real
        t_move, sweep = robot.transfer(self._q, nom.q_plan[a], self.qlim, self.v, self.a)
        joint_motion = float(np.abs(sweep[0] - self._q).sum())  # rad, between passes
        t_scan = float(nom.scan_time[a])
        self._q = sweep[-1].copy()
        self._used[a] = True
        self._step += 1
        self._covered |= real.visibility[a]
        self._covered_nom |= nom.visibility[a]
        cov = self.coverages()
        c_old, c_new = self._c, cov["coverage"]
        self._c, self._c_nom = c_new, cov["nominal_coverage"]
        self._t_move += t_move
        self._joint_motion += joint_motion
        self._t_scan += t_scan

        success = c_new >= self.target
        truncated = not success and (self._step >= self.k_max or not self.action_masks().any())
        parts_r = {
            "shaping": self.alpha * (self.discount * c_new - c_old),
            "move": -self.beta * t_move,
            "scan": -self.gamma_t * t_scan,
            "bonus": self.bonus if success else 0.0,
            "home": 0.0,
            "terminal_potential": 0.0,
        }
        t_home = 0.0
        if success or truncated:
            t_home = float(robot.movej_time(self.q_home - self._q, self.v, self.a))
            self._t_home = t_home
            self._joint_motion += float(np.abs(self.q_home - self._q).sum())
            parts_r["home"] = -self.beta * t_home
            if self.phi_zero:
                parts_r["terminal_potential"] = -self.alpha * self.discount * c_new
            self._done = True
        reward = self.scale * sum(parts_r.values())
        info = {
            "pass": a, "t_move": t_move, "t_scan": t_scan, "t_home": t_home, "joint_motion": joint_motion,
            "delta_coverage": c_new - c_old, "reward_parts": parts_r, **cov,
            "coverage_gap_nominal_minus_real": cov["nominal_coverage"] - c_new,
            "cycle_time": self.cycle_time, "total_joint_motion": self._joint_motion,
            "success": success, "action_mask": self.action_masks(),
        }
        return self._obs(), reward, success, truncated, info

    @property
    def cycle_time(self) -> float:
        """Arm motion + scanning + return home (once the episode has ended), seconds (D-024)."""
        return self._t_move + self._t_scan + self._t_home

    def state(self) -> dict:
        """Full internal state, including privileged data (for tests and debugging only)."""
        return {
            "variant": self.data.variant_id, "covered": self._covered.copy(), "covered_nominal": self._covered_nom.copy(),
            "used": self._used.copy(), "q": self._q.copy(), "step": self._step, "done": self._done,
            "t_move": self._t_move, "t_scan": self._t_scan, "t_home": self._t_home, "real": self.data.real,
        }

    def finish(self) -> tuple[float, dict]:
        """End the episode early (truncation by the caller, e.g. a sequence ran out); returns home."""
        if self._done:
            raise RuntimeError("episode is already over")
        t_home = float(robot.movej_time(self.q_home - self._q, self.v, self.a))
        self._t_home = t_home
        self._joint_motion += float(np.abs(self.q_home - self._q).sum())
        parts_r = {"home": -self.beta * t_home, "terminal_potential": -self.alpha * self.discount * self._c if self.phi_zero else 0.0}
        self._done = True
        return self.scale * sum(parts_r.values()), {
            "t_home": t_home, "reward_parts": parts_r, "cycle_time": self.cycle_time,
            "total_joint_motion": self._joint_motion,
        }


Policy = Callable[[dict, NominalModel, np.ndarray], "int | None"]


def _run(env: NBVEnv, variant_id: str, plan: list[int], follow: Policy | None, follow_is_repair: bool) -> dict:
    obs, info = env.reset(options={"variant": variant_id})
    steps, total_reward, done = [], 0.0, False
    keys = ("pass", "t_move", "t_scan", "t_home", "joint_motion", "coverage", "coverage_nsc", "coverage_raw",
            "nominal_coverage", "cycle_time", "total_joint_motion")

    def run(a: int, is_repair: bool) -> bool:
        nonlocal obs, total_reward
        obs, r, term, trunc, info = env.step(a)
        total_reward += r
        steps.append({k: info[k] for k in keys} | {"repair": is_repair})
        return term or trunc

    for a in plan:
        if done := run(a, False):
            break
    n_planned = len(steps)
    while not done and follow is not None:
        a = follow(obs, env.nominal, env.action_masks())
        if a is None:
            break
        done = run(a, follow_is_repair)
    if not done:
        r, fin = env.finish()
        total_reward += r
        if steps:
            steps[-1].update(t_home=fin["t_home"], cycle_time=fin["cycle_time"],
                             total_joint_motion=fin["total_joint_motion"])
    success = bool(steps) and steps[-1]["coverage"] >= env.target
    return {
        "variant": variant_id, "env_version": ENV_VERSION, "n_candidates": env.n_candidates,
        "steps": steps, "passes_executed": len(steps), "passes_ignored": len(plan) - n_planned,
        "repair_passes": len(steps) - n_planned if follow_is_repair else 0,
        "success": success, "censored": not success,
        "passes_to_target": len(steps) if success else None,
        "cycle_time": env.cycle_time, "total_joint_motion": steps[-1]["total_joint_motion"] if steps else 0.0,
        "return": total_reward,
    }


def evaluate_sequence(env: NBVEnv, variant_id: str, passes: list[int], repair: Policy | None = None) -> dict:
    """Run a pass sequence through the environment (the only path for reported cycle times, D-024).

    Stops at success or truncation; passes after that are ignored. If the sequence ends first and a
    `repair` policy is given (open-loop plans, D-032), it adds passes from the observation and the
    nominal model until success, K_max, or until it returns None; otherwise the episode is finished
    there (censored). Raises ValueError on a masked pass.
    """
    return _run(env, variant_id, passes, repair, True)


def evaluate_policy(env: NBVEnv, variant_id: str, make_policy: Callable[[NBVEnv], Policy]) -> dict:
    """Run a closed-loop policy until success, truncation, or until it returns None (censored).

    `make_policy(env)` is called after reset, so privileged baselines (greedy-oracle) can bind the
    episode's real part explicitly; nominal baselines must not use `env` beyond configuration.
    """
    env.reset(options={"variant": variant_id})
    policy = make_policy(env)
    return _run(env, variant_id, [], policy, False)
