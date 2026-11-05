"""Robot–part and robot–table geometry, clearance and diagnostic functions (D-042).

Since ENV v4, robot.plan_passes uses these clearances to mask colliding scan passes.

The UR5 is approximated by capsules along its links and the sensor head by a box. The DH model
collapses the lateral joint offsets of the real arm onto the joint-2/3/4 axis line; the capsules
put them back (shoulder +135.85 mm, elbow -119.7 mm along that axis, so that the sum with the
wrist-1 offset gives the DH d4 = 109.15 mm). Clearance = signed distance of the capsule axis to the
part surface minus the radius (negative = penetration); for the box, the smallest signed distance
of points on its faces, and negative if part vertices lie inside it. Distances to the part come
from Open3D's RaycastingScene (the mesh must be closed); the table is the plane z = 0.
All lengths in mm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
import roboticstoolbox as rtb
import trimesh


@dataclass(frozen=True)
class ArmGeometry:
    capsules: list[dict]      # name, radius, start / end: (frame index, offset along the joint-2 axis in mm)
    box_lo: np.ndarray        # (3,) sensor frame, mm
    box_hi: np.ndarray
    step: float               # sampling step along capsule axes and on box faces, mm
    table_exempt: tuple       # bodies mounted on the table (excluded from the table check)

    @classmethod
    def from_config(cls, cfg: dict) -> ArmGeometry:
        b = cfg["sensor_box_mm"]
        lo, hi = (np.array([b[k][i] for k in ("x", "y", "z")], float) for i in (0, 1))
        return cls(cfg["capsules"], lo, hi, float(cfg["sample_step_mm"]), tuple(cfg["table_exempt"]))

    def box_points(self) -> np.ndarray:
        """(M, 3) points on the box faces in the sensor frame, spacing <= step."""
        axes = [np.linspace(lo, hi, max(int(np.ceil((hi - lo) / self.step)) + 1, 2)) for lo, hi in zip(self.box_lo, self.box_hi)]
        pts = []
        for k in range(3):
            i, j = [a for a in range(3) if a != k]
            u, v = np.meshgrid(axes[i], axes[j], indexing="ij")
            for c in (self.box_lo[k], self.box_hi[k]):
                p = np.empty((u.size, 3))
                p[:, i], p[:, j], p[:, k] = u.ravel(), v.ravel(), c
                pts.append(p)
        return np.vstack(pts)


def capsule_axes(robot: rtb.DHRobot, q: np.ndarray, geom: ArmGeometry) -> list[tuple[str, float, np.ndarray, np.ndarray]]:
    """(name, radius, start, end) per capsule in world mm for joint configuration q."""
    T = robot.fkine_all(q)
    origins = T.t * 1000.0
    lateral = T[1].R[:, 2]  # joint-2 axis; joints 2-4 are parallel
    out = []
    for c in geom.capsules:
        (fa, oa), (fb, ob) = c["start"], c["end"]
        out.append((c["name"], float(c["radius"]), origins[fa] + oa * lateral, origins[fb] + ob * lateral))
    return out


def _segment_points(a: np.ndarray, b: np.ndarray, step: float) -> np.ndarray:
    n = max(int(np.ceil(np.linalg.norm(b - a) / step)) + 1, 2)
    return a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)


class PartDistance:
    """Signed distance to a closed part mesh (positive outside) and a vertex-in-box test."""

    def __init__(self, mesh: trimesh.Trimesh):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(mesh.vertices.astype(np.float32)),
                                 o3d.core.Tensor(mesh.faces.astype(np.uint32)))
        self.vertices = mesh.vertices

    def signed(self, points: np.ndarray) -> np.ndarray:
        return self.scene.compute_signed_distance(o3d.core.Tensor(points.astype(np.float32))).numpy().astype(float)

    def box_penetration(self, T_sensor_mm: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
        """Depth (mm, >= 0) of the deepest part vertex inside the box; 0 if none."""
        local = (self.vertices - T_sensor_mm[:3, 3]) @ T_sensor_mm[:3, :3]
        inside = np.all((local >= lo) & (local <= hi), axis=1)
        if not inside.any():
            return 0.0
        p = local[inside]
        return float(np.minimum(p - lo, hi - p).min(1).max())


def pose_clearance(robot: rtb.DHRobot, q: np.ndarray, part: PartDistance, geom: ArmGeometry,
                   box_local: np.ndarray, table_z: float = 0.0) -> dict[str, tuple[float, float]]:
    """{body: (clearance to the part, clearance to the table)} in mm for one joint configuration."""
    out = {}
    skip = {c["name"]: float(c.get("table_skip_mm", 0.0)) for c in geom.capsules}
    for name, r, a, b in capsule_axes(robot, q, geom):
        pts = _segment_points(a, b, geom.step)
        far = np.linalg.norm(pts - a, axis=1) >= skip[name]  # skip the part inside a fixed-height housing
        table = np.inf if name in geom.table_exempt else float(pts[far, 2].min() - table_z - r)
        out[name] = (float(part.signed(pts).min() - r), table)
    T = robot.fkine(q).A.copy()
    T[:3, 3] *= 1000.0
    pts = box_local @ T[:3, :3].T + T[:3, 3]
    corners = np.array([[x, y, z] for x in (geom.box_lo[0], geom.box_hi[0]) for y in (geom.box_lo[1], geom.box_hi[1])
                        for z in (geom.box_lo[2], geom.box_hi[2])]) @ T[:3, :3].T + T[:3, 3]
    c_part = float(part.signed(pts).min())
    depth = part.box_penetration(T, geom.box_lo, geom.box_hi)
    if depth > 0:
        c_part = min(c_part, -depth)
    out["sensor_head"] = (c_part, float(corners[:, 2].min() - table_z))
    return out
