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
    assert plan.ptp_pi.shape == (len(plan.reachable),) * 2
    assert np.all(np.diag(plan.ptp_pi) == 0.0)
    assert np.all(plan.ptp_pi >= 0.0)  # inf for unreachable passes
    assert not np.isnan(plan.ptp_pi).any()


def test_some_but_not_all_reachable(plan):
    assert 0 < plan.reachable.sum() < len(plan.reachable)


def test_reachable_sweeps_are_continuous(rc, plan):
    q = plan.q[plan.reachable]
    assert np.abs(np.diff(q, axis=1)).max() <= np.radians(rc["ik"]["max_step_deg"]) + 1e-9


def test_wrapped_ptp_is_a_lower_bound(rc, plan):
    m = rc["motion"]
    r = plan.reachable
    raw = robot.movej_time(plan.q[r, -1][:, None, :] - plan.q[r, 0][None, :, :], m["v_rad_s"], m["a_rad_s2"])
    np.fill_diagonal(raw, 0.0)
    assert np.all(np.isfinite(plan.ptp_pi[np.ix_(r, r)])) and np.all(np.isinf(plan.ptp_pi[~r][:, r]))
    assert np.all(plan.ptp_pi[np.ix_(r, r)] <= raw + 1e-12)
    assert np.all(np.diag(plan.ptp_pi) == 0.0)
    assert np.degrees(plan.leading_dq(wrapped=True)).max() <= 180.0 + 1e-9


def test_ptp_matches_movej_on_reachable(rc, plan):
    i, j = np.flatnonzero(plan.reachable)[:2]
    m = rc["motion"]
    expected = robot.movej_time(robot.wrap_pi(plan.q[j, 0] - plan.q[i, -1]), m["v_rad_s"], m["a_rad_s2"])
    assert np.isclose(plan.ptp_pi[i, j], expected)


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


def test_wrap_to_home_moves_midpoint_close_to_home():
    lim = np.radians(360.0)
    qlim = np.array([[-lim] * 6, [lim] * 6])
    home = np.radians([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])
    sweep = home + np.radians([350.0, 0, 0, 0, 0, -340.0]) + np.linspace(0, 0.1, 5)[:, None]
    w = robot.wrap_to_home(sweep, home, qlim)
    assert np.all(np.abs(w[2] - home) <= np.pi + 1e-9)
    # Joints within the 270 deg threshold are not touched, even when 2*pi closer exists.
    near_pi = home + np.radians([200.0, 0, 0, 0, 0, -190.0]) + np.zeros((5, 1))
    assert np.allclose(robot.wrap_to_home(near_pi, home, qlim), near_pi)
    assert np.allclose(np.diff(w, axis=0), np.diff(sweep, axis=0))  # continuity kept
    assert np.allclose(np.mod(w - sweep + np.pi, 2 * np.pi) - np.pi, 0.0)  # same poses


def test_wrap_to_home_respects_limits():
    lim = np.radians(360.0)
    qlim = np.array([[-lim] * 6, [lim] * 6])
    home = np.zeros(6)
    sweep = np.zeros((3, 6))
    # Midpoint 300 deg from home triggers the guard; -360 would bring it to -60 deg but push the
    # start to -560 deg < -360 deg, and +360 moves it away, so the sweep is left as is.
    sweep[:, 0] = np.radians([-200.0, 300.0, 350.0])
    w = robot.wrap_to_home(sweep, home, qlim)
    assert np.allclose(w, sweep)


def test_reachable_midpoints_are_wrapped_towards_home(rc, plan):
    home = robot.q_home(rc)
    mid = plan.q[plan.reachable, plan.q.shape[1] // 2]
    lim = np.radians(rc["joint_limit_deg"])
    far = np.abs(mid - home) > 1.5 * np.pi
    shifted_closer = far & (np.abs(mid - 2 * np.pi * np.sign(mid - home) - home) < np.abs(mid - home))
    shifted = plan.q[plan.reachable] - 2 * np.pi * np.sign(mid - home)[:, None, :]
    shifted_closer &= (shifted.max(1) <= lim) & (shifted.min(1) >= -lim)
    assert not shifted_closer.any()


@pytest.fixture(scope="module")
def box_mesh():
    m = trimesh.creation.box(extents=[200.0, 120.0, 80.0])
    m.apply_translation([0.0, 0.0, 40.0])
    return m


def test_collision_check_masks_close_passes(passes, box_mesh):
    rc2 = parts.load_yaml("configs/robot_v2.yaml")
    plan = robot.plan_passes(passes, rc2, 100.0, part_mesh=box_mesh)
    margin = rc2["collision"]["min_clearance_mm"]
    assert np.all(plan.ik_reachable >= plan.reachable)
    assert np.all(plan.min_clearance[plan.reachable] >= margin)
    masked = plan.ik_reachable & ~plan.reachable
    assert np.all(plan.min_clearance[masked] < margin)
    assert np.all(np.isnan(plan.q[masked])) and np.all(np.isinf(plan.ptp_pi[masked][:, plan.reachable]))
    assert np.all(np.isnan(plan.min_clearance[~plan.ik_reachable]))


def test_collision_config_needs_the_part_mesh(passes):
    with pytest.raises(ValueError):
        robot.plan_passes(passes, parts.load_yaml("configs/robot_v2.yaml"), 100.0)


def test_without_collision_section_nothing_is_masked(plan):
    assert np.array_equal(plan.reachable, plan.ik_reachable) and np.all(np.isinf(plan.min_clearance[plan.reachable]))
