import numpy as np
import pytest
import trimesh

from nbv import candidates, parts


@pytest.fixture(scope="module")
def cfg():
    return parts.load_yaml("configs/candidates_v1.yaml"), parts.load_yaml("configs/sensor_v1.yaml")


@pytest.fixture(scope="module")
def mesh():
    m = trimesh.creation.box(extents=[200.0, 120.0, 80.0])
    m.apply_translation([0.0, 0.0, 40.0])  # bbox bottom center at the origin, as in parts.normalize
    return m


@pytest.fixture(scope="module", params=["n_candidates", "n_candidates_sensitivity"])
def passes(request, cfg, mesh):
    cc, sensor = cfg
    return candidates.make_passes(
        mesh.vertices, cc[request.param], cc["min_elevation_deg"], candidates.standoff_offset(sensor), cc["sweep_margin_mm"]
    )


def test_counts(cfg, passes):
    cc, _ = cfg
    assert passes.n in (cc["n_candidates"], cc["n_candidates_sensitivity"])
    for a in (passes.direction, passes.view, passes.sweep, passes.midpoint):
        assert a.shape == (passes.n, 3)


def test_unit_vectors(passes):
    for a in (passes.direction, passes.view, passes.sweep):
        assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.allclose(passes.view, -passes.direction)


def test_elevation_limit(cfg, passes):
    cc, _ = cfg
    el = np.degrees(np.arcsin(passes.direction[:, 2]))
    assert el.min() >= cc["min_elevation_deg"] - 1e-9
    assert el.max() <= 90.0


def test_sweep_perpendicular_to_view(passes):
    assert np.allclose((passes.sweep * passes.view).sum(1), 0.0, atol=1e-12)


def test_sensor_rotation_is_proper(passes):
    R = passes.rotation
    assert np.allclose(np.einsum("nji,njk->nik", R, R), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(R), 1.0)
    assert np.allclose(R[:, :, 2], passes.view)


def test_sweep_covers_part_with_margin(cfg, mesh, passes):
    cc, _ = cfg
    proj = (mesh.vertices - passes.center) @ passes.sweep.T
    half = passes.length / 2 - cc["sweep_margin_mm"]
    assert np.all(np.abs(proj).max(0) <= half + 1e-9)


def test_standoff_keeps_part_beyond_near_dof_limit(cfg, mesh, passes):
    _, sensor = cfg
    # Distance along the view axis from every sensor sample to every part vertex >= near DOF limit.
    near = candidates.standoff_offset(sensor)
    for pos in passes.samples(5).transpose(1, 0, 2):
        depth = ((mesh.vertices[:, None, :] - pos[None]) * passes.view[None]).sum(-1)
        assert depth.min() >= near - 1e-9


def test_zenith_has_defined_sweep():
    s = candidates.sweep_directions(np.array([[0.0, 0.0, -1.0]]))
    assert np.allclose(s, [[1.0, 0.0, 0.0]])


def test_deterministic(cfg, mesh):
    cc, sensor = cfg
    a = candidates.make_passes(mesh.vertices, 48, 10.0, candidates.standoff_offset(sensor), 20.0)
    b = candidates.make_passes(mesh.vertices, 48, 10.0, candidates.standoff_offset(sensor), 20.0)
    assert np.array_equal(a.direction, b.direction) and np.array_equal(a.length, b.length)
