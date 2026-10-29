"""Dual-occlusion ray casting (laser + camera) and per-part visibility cache.

Analytic sweep: the laser plane (sensor y-z plane) is normal to the sweep direction s, so a
sweep crosses every surface point p exactly once, when the sensor origin is at
O(t*) = midpoint + t* s with t* = (p - midpoint) . s. At that instant p is measured if

    geom         |t*| <= L/2, near <= depth <= far, |lateral| <= depth * tan(fan half-angle)
    laser_angle  angle(laser ray, normal) <= max_laser_incidence
    laser_los    segment p -> laser source O(t*) unobstructed
    cam_angle    angle(camera ray, normal) <= max_camera_angle
    cam_los      segment p -> camera C(t*) = O(t*) + side * b * s unobstructed

with b = working_distance * tan(triangulation angle) (D-019, D-025). The camera's field of view
is assumed to cover the laser line over the depth of field. Line-of-sight bits are only computed
where `geom` and the matching angle test pass; elsewhere they are False.
All lengths in mm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from nbv.candidates import ScanPasses

LAYERS = ("geom", "laser_angle", "laser_los", "cam_angle", "cam_los")


# --- Surface samples ---------------------------------------------------------


@dataclass(frozen=True)
class SurfaceSamples:
    """Area-weighted samples as (face index, barycentric coordinates), valid on every variant of a part."""

    face: np.ndarray  # (N,) int
    bary: np.ndarray  # (N, 3)

    def on(self, mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
        """Points and (face) normals of the samples on `mesh` (same topology as the base mesh)."""
        p = np.einsum("ij,ijk->ik", self.bary, mesh.triangles[self.face])
        return p, mesh.face_normals[self.face]


def sample_surface(mesh: trimesh.Trimesh, n: int, seed: int | list[int]) -> SurfaceSamples:
    """Area-weighted samples; `seed` may be a sequence, e.g. [sample_seed, file_id]."""
    rng = np.random.default_rng(seed)
    area = mesh.area_faces
    face = rng.choice(len(area), size=n, p=area / area.sum())
    return SurfaceSamples(face, rng.dirichlet(np.ones(3), size=n))


# --- Sensor geometry ---------------------------------------------------------


@dataclass(frozen=True)
class SensorGeometry:
    near: float
    far: float
    tan_fan: float           # tan(fan half-angle)
    baseline: float          # camera offset along the sweep axis
    camera_sides: tuple      # +1 / -1 per camera
    cos_laser: float         # cos(max laser incidence)
    cos_camera: float        # cos(max camera angle)

    @classmethod
    def from_config(cls, sensor: dict, vis: dict) -> SensorGeometry:
        wd, half_dof = sensor["working_distance_mm"], sensor["depth_of_field_mm"] / 2
        return cls(
            near=wd - half_dof,
            far=wd + half_dof,
            tan_fan=(sensor["line_width_mm"] / 2) / wd,
            baseline=wd * np.tan(np.radians(sensor["triangulation_angle_deg"])),
            camera_sides=tuple(vis["camera_sides"]),
            cos_laser=float(np.cos(np.radians(vis["max_laser_incidence_deg"]))),
            cos_camera=float(np.cos(np.radians(vis["max_camera_angle_deg"]))),
        )


# --- Visibility layers -------------------------------------------------------


@dataclass(frozen=True)
class Visibility:
    """Per (pass, sample) layers; camera layers have a leading camera axis (C, P, N)."""

    geom: np.ndarray         # (P, N) bool
    laser_angle: np.ndarray  # (P, N) bool
    laser_los: np.ndarray    # (P, N) bool
    cam_angle: np.ndarray    # (C, P, N) bool
    cam_los: np.ndarray      # (C, P, N) bool

    def laser_ok(self) -> np.ndarray:
        return self.geom & self.laser_angle & self.laser_los

    def camera_ok(self) -> np.ndarray:
        """(P, N): seen by at least one camera (geom + camera angle + camera line of sight)."""
        return self.geom & (self.cam_angle & self.cam_los).any(0)

    def measured(self) -> np.ndarray:
        """(P, N) dual-occlusion model: laser reaches the point and a camera sees it."""
        return self.laser_ok() & (self.cam_angle & self.cam_los).any(0)

    def camera_only(self) -> np.ndarray:
        """(P, N) pure camera simulator: laser occlusion and incidence not modeled (ablation, D-026)."""
        return self.camera_ok()

    def save(self, path: str | Path) -> None:
        n = self.geom.shape[-1]
        np.savez_compressed(path, n_samples=n, **{k: np.packbits(getattr(self, k), axis=-1) for k in LAYERS})

    @classmethod
    def load(cls, path: str | Path) -> Visibility:
        z = np.load(path)
        n = int(z["n_samples"])
        return cls(**{k: np.unpackbits(z[k], axis=-1, count=n).astype(bool) for k in LAYERS})


def make_scene(mesh: trimesh.Trimesh, table_z: float | None) -> o3d.t.geometry.RaycastingScene:
    """Part mesh plus, optionally, a large table quad at z = table_z as an occluder."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(mesh.vertices.astype(np.float32)), o3d.core.Tensor(mesh.faces.astype(np.uint32))
    )
    if table_z is not None:
        r = 1e4
        v = np.array([[-r, -r, table_z], [r, -r, table_z], [r, r, table_z], [-r, r, table_z]], np.float32)
        scene.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(np.array([[0, 1, 2], [0, 2, 3]], np.uint32)))
    return scene


def _unoccluded(scene, origins: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if len(origins) == 0:
        return np.zeros(0, bool)
    rays = np.hstack([origins, targets - origins]).astype(np.float32)  # tfar = 1 reaches the target
    return ~scene.test_occlusions(o3d.core.Tensor(rays), tnear=0.0, tfar=1.0).numpy().astype(bool)


def compute_visibility(
    scene, points: np.ndarray, normals: np.ndarray, passes: ScanPasses, g: SensorGeometry, ray_offset: float
) -> Visibility:
    P, N, C = passes.n, len(points), len(g.camera_sides)
    out = {k: np.zeros((P, N), bool) for k in ("geom", "laser_angle", "laser_los")}
    cam_angle = np.zeros((C, P, N), bool)
    cam_los = np.zeros((C, P, N), bool)
    origin = points + ray_offset * normals
    lateral_axis = np.cross(passes.view, passes.sweep)
    for i in range(P):
        s, d = passes.sweep[i], passes.view[i]
        t = (points - passes.midpoint[i]) @ s
        source = passes.midpoint[i] + t[:, None] * s
        rel = points - source
        depth, lateral = rel @ d, rel @ lateral_axis[i]
        geom = (np.abs(t) <= passes.length[i] / 2) & (depth >= g.near) & (depth <= g.far)
        geom &= np.abs(lateral) <= depth * g.tan_fan
        out["geom"][i] = geom
        to_src = source - points
        la = geom & ((to_src * normals).sum(1) >= g.cos_laser * np.linalg.norm(to_src, axis=1))
        out["laser_angle"][i] = la
        idx = np.flatnonzero(la)
        out["laser_los"][i, idx] = _unoccluded(scene, origin[idx], source[idx])
        for c, side in enumerate(g.camera_sides):
            cam = source + side * g.baseline * s
            to_cam = cam - points
            ca = geom & ((to_cam * normals).sum(1) >= g.cos_camera * np.linalg.norm(to_cam, axis=1))
            cam_angle[c, i] = ca
            idx = np.flatnonzero(ca)
            cam_los[c, i, idx] = _unoccluded(scene, origin[idx], cam[idx])
    return Visibility(out["geom"], out["laser_angle"], out["laser_los"], cam_angle, cam_los)


# --- Coverage summaries ------------------------------------------------------


def union(matrix: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """(N,) samples covered by any pass (optionally only `rows`)."""
    m = matrix if rows is None else matrix[rows]
    return m.any(0)


def coverage_report(vis: Visibility, reachable: np.ndarray) -> dict:
    """Per-variant ratios (D-023): reachable vs all-pass unions and occlusion losses (fractions of samples)."""
    meas, cam, las = vis.measured(), vis.camera_only(), vis.laser_ok()
    u_all, u_reach = union(meas), union(meas, reachable)
    u_cam, u_las = union(cam), union(las)
    n = meas.shape[1]
    return {
        "union_all_over_samples": u_all.sum() / n,
        "observable_reachable_over_all": u_reach.sum() / max(u_all.sum(), 1),
        "laser_occlusion_loss": (u_cam & ~u_all).sum() / n,
        "camera_occlusion_loss": (u_las & ~u_all).sum() / n,
    }
