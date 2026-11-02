import numpy as np
import pytest
import trimesh

from nbv import collision, parts, robot


@pytest.fixture(scope="module")
def setup():
    rc = parts.load_yaml("configs/robot_v1.yaml")
    geom = collision.ArmGeometry.from_config(parts.load_yaml("configs/collision_v1.yaml"))
    return robot.make_robot(rc), robot.q_home(rc), geom, geom.box_points()


def cube_at(center, size=30.0):
    m = trimesh.creation.box(extents=(size, size, size))
    m.apply_translation(center)
    return collision.PartDistance(m)


def test_box_points_lie_on_the_faces(setup):
    *_, geom, box = setup
    assert np.all(box >= geom.box_lo - 1e-9) and np.all(box <= geom.box_hi + 1e-9)
    on_face = np.isclose(box, geom.box_lo) | np.isclose(box, geom.box_hi)
    assert on_face.any(1).all()
    spacing = np.diff(np.unique(box[:, 0]))
    assert spacing.max() <= geom.step + 1e-9


def test_signed_distance_sign():
    part = cube_at(np.zeros(3), 40.0)
    d = part.signed(np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]))
    assert d[0] < 0 and np.isclose(d[1], 30.0, atol=0.1)


def test_far_part_gives_positive_clearance(setup):
    arm, q, geom, box = setup
    out = collision.pose_clearance(arm, q, cube_at([3000.0, 0.0, 500.0]), geom, box)
    assert all(c > 1000 for c, _ in out.values())
    assert set(out) == {c["name"] for c in geom.capsules} | {"sensor_head"}


def test_part_around_a_link_is_a_collision(setup):
    arm, q, geom, box = setup
    name, r, a, b = next(c for c in collision.capsule_axes(arm, q, geom) if c[0] == "upper_arm")
    out = collision.pose_clearance(arm, q, cube_at((a + b) / 2, 20.0), geom, box)
    assert out["upper_arm"][0] < -r + 15  # axis inside the cube
    assert out["wrist_3"][0] > 0


def test_part_inside_the_sensor_head_is_a_collision(setup):
    arm, q, geom, box = setup
    T = arm.fkine(q).A
    center = T[:3, 3] * 1000 + T[:3, :3] @ np.array([40.0, 0.0, -60.0])  # inside the head, away from its faces
    out = collision.pose_clearance(arm, q, cube_at(center, 10.0), geom, box)
    assert out["sensor_head"][0] < 0


def test_lateral_offsets_follow_the_dh_d4(setup):
    arm, q, geom, _ = setup
    axes = {n: (a, b) for n, _, a, b in collision.capsule_axes(arm, q, geom)}
    T = arm.fkine_all(q).t * 1000
    assert np.allclose(axes["wrist_1"][1], T[4])  # wrist 1 ends at DH frame 4 (d4 = 109.15 mm)
    assert np.isclose(np.linalg.norm(axes["shoulder"][1] - axes["shoulder"][0]), 135.85)


def test_shoulder_housing_is_not_table_checked(setup):
    arm, q, geom, box = setup
    out = collision.pose_clearance(arm, q, cube_at([3000.0, 0.0, 500.0]), geom, box)
    assert out["base"][1] == out["shoulder"][1] == np.inf
    assert out["upper_arm"][1] > 80  # upright at q_home; the fixed-height housing part is skipped
