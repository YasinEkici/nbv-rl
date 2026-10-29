"""Robot model: IK reachability mask and joint-space PTP time matrix.

UR5 from the Robotics Toolbox DH model (rtb.models.DH.UR5; the URDF model downloads data at
first use). Configs are in mm/deg; the robot works in m/rad, converted only here.

`ik_LM` on the DH model does not apply `robot.tool` consistently, so IK is always solved for the
flange pose in the robot base frame on a bare model (no base, no tool); forward kinematics uses the
placed model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

from nbv.candidates import ScanPasses


def make_robot(cfg: dict) -> rtb.DHRobot:
    """UR5 placed in the cell, with the sensor origin as the tool frame (for forward kinematics)."""
    r = _ur5(cfg)
    x, y, z = np.asarray(cfg["base_xyz_mm"]) / 1000.0
    r.base = SE3(x, y, z) * SE3.Rz(np.radians(cfg["base_yaw_deg"]))
    r.tool = SE3(0.0, 0.0, cfg["tcp_offset_mm"] / 1000.0)
    return r


def _ur5(cfg: dict) -> rtb.DHRobot:
    r = rtb.models.DH.UR5()
    lim = np.radians(cfg["joint_limit_deg"])
    r.qlim = np.array([[-lim] * r.n, [lim] * r.n])
    return r


@dataclass(frozen=True)
class IKSolver:
    """Sensor-pose IK: world sensor pose -> flange pose in the base frame -> bare-model ik_LM."""

    bare: rtb.DHRobot
    base_inv: np.ndarray
    tool_inv: np.ndarray
    ilimit: int
    tol: float

    @classmethod
    def from_robot(cls, robot: rtb.DHRobot, cfg: dict) -> IKSolver:
        ik = cfg["ik"]
        return cls(_ur5(cfg), np.linalg.inv(robot.base.A), np.linalg.inv(robot.tool.A), ik["ilimit"], ik["tol"])

    def __call__(self, T_sensor: np.ndarray, q0: np.ndarray) -> np.ndarray | None:
        T = self.base_inv @ T_sensor @ self.tool_inv
        # One search per call: ik_LM's internal restarts (slimit > 1) use an unseeded RNG.
        sol = self.bare.ik_LM(T, q0=q0, ilimit=self.ilimit, slimit=1, tol=self.tol, joint_limits=True)
        return np.asarray(sol.q) if sol.success else None


def q_home(cfg: dict) -> np.ndarray:
    return np.radians(cfg["q_home_deg"])


def sensor_pose(position_mm: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """4x4 sensor pose in m from a position in mm and a rotation matrix."""
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position_mm / 1000.0
    return T


def movej_time(dq: np.ndarray, v: float, a: float) -> np.ndarray:
    """URScript movej duration for joint displacement(s) `dq` (..., 6), trapezoid on the leading axis.

    t = d/v + v/a if d >= v^2/a, else 2*sqrt(d/a), with d = max_k |dq_k|.
    """
    d = np.abs(dq).max(-1)
    return np.where(d >= v * v / a, d / v + v / a, 2.0 * np.sqrt(d / a))


def clears_table(robot: rtb.DHRobot, q: np.ndarray, clearance_m: float) -> bool:
    """All DH frame origins after the base, and the sensor origin, are above the table plane."""
    z = robot.fkine_all(q).t[1:, 2]
    return bool(z.min() >= clearance_m and robot.fkine(q).t[2] >= clearance_m)


def solve_sweep(
    robot: rtb.DHRobot, solve: IKSolver, poses: np.ndarray, home: np.ndarray, starts: np.ndarray, ik: dict
) -> np.ndarray | None:
    """Joint configurations (k, 6) along one sweep, or None if the pass is unreachable.

    The first pose is solved from q_home and from each seeded random start; the solution closest
    to q_home is kept. Later poses are seeded with the previous solution so the sweep stays on one
    IK branch; a pass is rejected if any joint moves more than max_step between samples or the
    arm/sensor goes below the table clearance.
    """
    sols = [q for q0 in (home, *starts) if (q := solve(poses[0], q0)) is not None]
    if not sols:
        return None
    q = [min(sols, key=lambda s: float(np.linalg.norm(s - home)))]
    max_step = np.radians(ik["max_step_deg"])
    for T in poses[1:]:
        nxt = solve(T, q[-1])
        if nxt is None or np.abs(nxt - q[-1]).max() > max_step:
            return None
        q.append(nxt)
    clearance = ik["table_clearance_mm"] / 1000.0
    if not all(clears_table(robot, qi, clearance) for qi in q):
        return None
    return np.array(q)


@dataclass(frozen=True)
class RobotPlan:
    reachable: np.ndarray   # (N,) bool
    q: np.ndarray           # (N, k, 6) rad; NaN where unreachable
    ptp: np.ndarray         # (N, N) s, end of pass i -> start of pass j; 0 on the diagonal, inf if unreachable
    from_home: np.ndarray   # (N,) s, q_home -> start of pass j
    to_home: np.ndarray     # (N,) s, end of pass i -> q_home
    scan_time: np.ndarray   # (N,) s, sweep length / scan speed


def plan_passes(passes: ScanPasses, cfg: dict, scan_speed_mm_s: float) -> RobotPlan:
    """IK for every pass and the PTP time matrix (D-021, D-022)."""
    robot = make_robot(cfg)
    solve = IKSolver.from_robot(robot, cfg)
    ik, motion = cfg["ik"], cfg["motion"]
    home = q_home(cfg)
    lo, hi = robot.qlim
    starts = np.random.default_rng(ik["seed"]).uniform(lo, hi, size=(ik["n_random_starts"], robot.n))
    k = ik["n_samples"]
    pos, rot = passes.samples(k), passes.rotation
    q = np.full((passes.n, k, robot.n), np.nan)
    for i in range(passes.n):
        poses = np.array([sensor_pose(p, rot[i]) for p in pos[i]])
        sol = solve_sweep(robot, solve, poses, home, starts, ik)
        if sol is not None:
            q[i] = sol
    reachable = ~np.isnan(q[:, 0, 0])
    v, a = motion["v_rad_s"], motion["a_rad_s2"]
    start, end = q[:, 0], q[:, -1]
    ptp = movej_time(end[:, None, :] - start[None, :, :], v, a)
    ptp[~reachable, :] = np.inf
    ptp[:, ~reachable] = np.inf
    np.fill_diagonal(ptp, 0.0)
    from_home = np.where(reachable, movej_time(start - home, v, a), np.inf)
    to_home = np.where(reachable, movej_time(end - home, v, a), np.inf)
    return RobotPlan(reachable, q, ptp, from_home, to_home, passes.length / scan_speed_mm_s)
