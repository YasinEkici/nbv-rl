import numpy as np
import pytest
import trimesh

from nbv import candidates, parts, robot


@pytest.fixture(scope="module")
def rc():
    return parts.load_yaml("configs/robot_v1.yaml")


@pytest.fixture(scope="module")
def passes():
    sensor = parts.load_yaml("configs/sensor_v1.yaml")
    cc = parts.load_yaml("configs/candidates_v1.yaml")
    m = trimesh.creation.box(extents=[200.0, 120.0, 80.0])
    m.apply_translation([0.0, 0.0, 40.0])
    return candidates.make_passes(
        m.vertices, cc["n_candidates"], cc["min_elevation_deg"], candidates.standoff_offset(sensor), cc["sweep_margin_mm"]
    )


@pytest.fixture(scope="module")
def plan(rc, passes):
    return robot.plan_passes(passes, rc, parts.load_yaml("configs/sensor_v1.yaml")["scan_speed_mm_s"])


def test_reachable_joint_solutions_reproduce_sensor_poses(rc, passes, plan):
    r = robot.make_robot(rc)
    k = rc["ik"]["n_samples"]
    pos, rot = passes.samples(k), passes.rotation
    for i in np.flatnonzero(plan.reachable):
        for j in range(k):
            # 1e-4 (0.1 mm / ~1e-4 rad): ik tol 1e-10 leaves ~1e-5 errors on some poses.
            assert np.allclose(r.fkine(plan.q[i, j]).A, robot.sensor_pose(pos[i, j], rot[i]), atol=1e-4)


def test_ik_solves_known_reachable_pose(rc):
    r = robot.make_robot(rc)
    q_target = robot.q_home(rc) + np.radians([20.0, 10.0, -15.0, 5.0, 10.0, 30.0])
    T = r.fkine(q_target).A
    q = robot.IKSolver.from_robot(r, rc)(T, robot.q_home(rc))
    assert q is not None
    assert np.allclose(r.fkine(q).A, T, atol=1e-6)


def test_ptp_diagonal_zero_and_non_negative(plan):
    assert plan.ptp.shape == (len(plan.reachable),) * 2
    assert np.all(np.diag(plan.ptp) == 0.0)
    assert np.all(plan.ptp >= 0.0)  # inf for unreachable passes
    assert not np.isnan(plan.ptp).any()


def test_some_but_not_all_reachable(plan):
    assert 0 < plan.reachable.sum() < len(plan.reachable)


def test_reachable_sweeps_are_continuous(rc, plan):
    q = plan.q[plan.reachable]
    assert np.abs(np.diff(q, axis=1)).max() <= np.radians(rc["ik"]["max_step_deg"]) + 1e-9


def test_ptp_matches_movej_on_reachable(rc, plan):
    i, j = np.flatnonzero(plan.reachable)[:2]
    m = rc["motion"]
    expected = robot.movej_time(plan.q[j, 0] - plan.q[i, -1], m["v_rad_s"], m["a_rad_s2"])
    assert np.isclose(plan.ptp[i, j], expected)


def test_movej_trapezoid():
    v, a = 1.05, 1.4
    d_switch = v * v / a
    long = np.array([2.0, 0.1, 0, 0, 0, 0])
    short = np.array([0.0, -0.3, 0, 0, 0, 0])
    assert np.isclose(robot.movej_time(long, v, a), 2.0 / v + v / a)
    assert np.isclose(robot.movej_time(short, v, a), 2 * np.sqrt(0.3 / a))
    # Continuous at the triangle/trapezoid switch point.
    eps = 1e-9
    t_lo = robot.movej_time(np.array([d_switch - eps, 0, 0, 0, 0, 0]), v, a)
    t_hi = robot.movej_time(np.array([d_switch + eps, 0, 0, 0, 0, 0]), v, a)
    assert np.isclose(t_lo, t_hi, atol=1e-6)


def test_plan_is_deterministic(rc, passes, plan):
    again = robot.plan_passes(passes, rc, parts.load_yaml("configs/sensor_v1.yaml")["scan_speed_mm_s"])
    assert np.array_equal(again.reachable, plan.reachable)
    assert np.array_equal(np.nan_to_num(again.q), np.nan_to_num(plan.q))
