import numpy as np
import pytest
import trimesh

from nbv import candidates, parts, visibility
from nbv.candidates import ScanPasses

ZENITH = np.array([[0.0, 0.0, 1.0]])


@pytest.fixture(scope="module")
def geometry():
    sensor = parts.load_yaml("configs/sensor_v1.yaml")
    vis = parts.load_yaml("configs/visibility_v1.yaml")
    return visibility.SensorGeometry.from_config(sensor, vis), vis


def zenith_pass(center_xy, top_z: float, depth: float, length: float) -> ScanPasses:
    """One pass looking straight down whose laser source sits `depth` above z = top_z."""
    view = -ZENITH
    sweep = candidates.sweep_directions(view)
    center = np.array([center_xy[0], center_xy[1], 0.0])
    return ScanPasses(center, ZENITH, view, sweep, np.array([top_z + depth]), np.array([length]))


def run(mesh, passes, geometry, n=4000, seed=0):
    g, vis = geometry
    s = visibility.sample_surface(mesh, n, seed)
    p, nrm = s.on(mesh)
    scene = visibility.make_scene(mesh, mesh.vertices[:, 2].min() - vis["table_offset_mm"])
    return p, nrm, visibility.compute_visibility(scene, p, nrm, passes, g, vis["ray_offset_mm"])


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    return m


def test_cube_top_pass_covers_top_face_only(geometry):
    cube = box([-30, -30, 0], [30, 30, 60])
    p, nrm, v = run(cube, zenith_pass((0, 0), 60.0, 170.0, 100.0), geometry)
    m = v.measured()[0]
    top, bottom, side = nrm[:, 2] > 0.99, nrm[:, 2] < -0.99, np.abs(nrm[:, 2]) < 0.01
    assert m[top].all()          # fan at 170 mm is 85 mm wide > 60 mm, camera angle ~29 deg
    assert not m[bottom].any()
    assert not m[side].any()     # laser incidence 90 deg


def test_overhang_camera_sees_where_laser_is_blocked(geometry):
    # Floor slab, wall at x = -40..-35, roof over x = -40..0 at z = 40..45.
    # Laser comes straight down: floor under the roof is blocked. The camera sits +93 mm along +x
    # (sweep axis), so it looks under the roof edge onto the floor at roughly x in (-13, 0).
    part = trimesh.util.concatenate([box([-40, -40, 0], [40, 40, 10]), box([-40, -40, 10], [-35, 40, 40]),
                                     box([-40, -40, 40], [0, 40, 45])])
    p, nrm, v = run(part, zenith_pass((0, 0), 45.0, 165.0, 140.0), geometry, n=20000)
    dual, cam = v.measured()[0], v.camera_only()[0]
    under_roof = (nrm[:, 2] > 0.99) & (np.abs(p[:, 2] - 10) < 1e-6) & (p[:, 0] > -12) & (p[:, 0] < -2)
    assert under_roof.sum() > 0
    assert cam[under_roof].all() and not dual[under_roof].any()
    assert dual.sum() < cam.sum()


def test_dual_is_subset_of_camera_only(geometry):
    mesh = trimesh.creation.annulus(r_min=20.0, r_max=60.0, height=50.0)
    mesh.apply_translation([0, 0, 25.0])
    sensor = parts.load_yaml("configs/sensor_v1.yaml")
    passes = candidates.make_passes(mesh.vertices, 48, 10.0, candidates.standoff_offset(sensor), 20.0)
    _, _, v = run(mesh, passes, geometry)
    assert not (v.measured() & ~v.camera_only()).any()
    assert v.measured().any(1).sum() > 0
    # Line-of-sight bits only where geometry and the matching angle test pass.
    assert not (v.laser_los & ~v.laser_angle).any() and not (v.cam_los & ~v.cam_angle).any()
    assert not (v.laser_angle & ~v.geom).any() and not (v.cam_angle & ~v.geom[None]).any()


def test_deterministic_and_roundtrip(geometry, tmp_path):
    mesh = trimesh.creation.annulus(r_min=20.0, r_max=60.0, height=50.0)
    mesh.apply_translation([0, 0, 25.0])
    sensor = parts.load_yaml("configs/sensor_v1.yaml")
    passes = candidates.make_passes(mesh.vertices, 8, 10.0, candidates.standoff_offset(sensor), 20.0)
    _, _, a = run(mesh, passes, geometry)
    _, _, b = run(mesh, passes, geometry)
    for k in visibility.LAYERS:
        assert np.array_equal(getattr(a, k), getattr(b, k))
    a.save(tmp_path / "v.npz")
    c = visibility.Visibility.load(tmp_path / "v.npz")
    for k in visibility.LAYERS:
        assert np.array_equal(getattr(a, k), getattr(c, k))


def test_samples_follow_the_variant_surface():
    cfg = parts.load_yaml("configs/parts_v1.yaml")
    g = cfg["geometry"]
    base = parts.subdivide(parts.normalize(trimesh.creation.annulus(20.0, 40.0, 30.0), g["longest_mm"]), g["max_edge_mm"])
    var = parts.make_variant(base, cfg["deviation"], parts.variant_rng(0, 1, 0))
    s = visibility.sample_surface(base, 2000, 0)
    assert np.array_equal(s.face, visibility.sample_surface(base, 2000, 0).face)
    p, _ = s.on(var)
    _, dist, _ = trimesh.proximity.closest_point(var, p)
    assert dist.max() < 1e-6


def test_coverage_report_ratios():
    geom = np.ones((2, 4), bool)
    v = visibility.Visibility(
        geom=geom,
        laser_angle=geom.copy(),
        laser_los=np.array([[1, 0, 0, 0], [0, 1, 0, 1]], bool),
        cam_angle=geom[None].copy(),
        cam_los=np.array([[[1, 1, 1, 0], [0, 1, 0, 0]]], bool),
    )
    r = visibility.coverage_report(v, reachable=np.array([True, False]))
    # measured: pass 0 -> {0}, pass 1 -> {1}; camera-only union {0,1,2}; laser-only union {0,1,3}
    assert r["union_all_over_samples"] == 0.5
    assert r["observable_reachable_over_all"] == 0.5
    assert r["laser_occlusion_loss"] == 0.25
    assert r["camera_occlusion_loss"] == 0.25


def test_coverage_against_reference():
    covered = np.array([1, 1, 0, 0, 1], bool)
    ref = np.array([1, 0, 1, 0, 1], bool)
    assert visibility.coverage(covered, ref) == 2 / 3
    assert visibility.coverage(covered, np.zeros(5, bool)) == 0.0
