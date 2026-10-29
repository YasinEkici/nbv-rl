import numpy as np
import pytest
import trimesh

from nbv import parts

CONFIG = "configs/parts_v1.yaml"


@pytest.fixture(scope="module")
def cfg():
    return parts.load_yaml(CONFIG)


@pytest.fixture(scope="module")
def base(cfg):
    # Synthetic watertight part with a through hole (genus 1); no downloaded data needed.
    g = cfg["geometry"]
    ring = trimesh.creation.annulus(r_min=20.0, r_max=40.0, height=30.0)
    out = parts.subdivide(parts.normalize(ring, g["longest_mm"]), g["max_edge_mm"])
    parts.check_mesh(out)
    return out


def test_same_seed_same_variant(cfg, base):
    a = parts.make_variant(base, cfg["deviation"], parts.variant_rng(0, 1, 0))
    b = parts.make_variant(base, cfg["deviation"], parts.variant_rng(0, 1, 0))
    assert parts.mesh_hash(a) == parts.mesh_hash(b)


def test_different_index_different_variant(cfg, base):
    a = parts.make_variant(base, cfg["deviation"], parts.variant_rng(0, 1, 0))
    b = parts.make_variant(base, cfg["deviation"], parts.variant_rng(0, 1, 1))
    assert parts.mesh_hash(a) != parts.mesh_hash(b)


def test_variant_is_valid_and_within_bounds(cfg, base):
    d = cfg["deviation"]
    rng = parts.variant_rng(0, 1, 0)
    v = parts.make_variant(base, d, rng)
    parts.check_mesh(v)
    assert np.array_equal(v.faces, base.faces)
    # Undo the per-axis scale (re-drawn from the same stream) and bound the normal displacement.
    rng = parts.variant_rng(0, 1, 0)
    disp = parts._fourier_noise(base.vertices, d["noise"], rng) + parts._dents(base, d["dents"], rng)
    scale = 1.0 + rng.uniform(-d["axis_scale"], d["axis_scale"], size=3)
    assert np.all(np.abs(scale - 1.0) <= d["axis_scale"])
    assert np.allclose(v.vertices / scale, base.vertices + disp[:, None] * base.vertex_normals)
    assert np.abs(disp).max() <= d["noise"]["amplitude_mm"] + d["dents"]["amplitude_mm"] + 1e-9


def test_normalize_sets_longest_side_and_table_contact():
    m = parts.normalize(trimesh.creation.box(extents=[3.0, 1.0, 2.0]), 200.0)
    assert np.isclose(m.extents.max(), 200.0)
    assert np.isclose(m.bounds[0, 2], 0.0)
    assert np.allclose(m.bounds[:, :2].sum(0), 0.0)


def test_mesh_hash_ignores_negative_zero():
    a = trimesh.Trimesh([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], process=False)
    b = trimesh.Trimesh([[-0.0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], process=False)
    assert parts.mesh_hash(a) == parts.mesh_hash(b)
