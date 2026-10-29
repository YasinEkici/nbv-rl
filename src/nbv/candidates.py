"""Candidate scan passes: Fibonacci view directions on an upper spherical cap and linear-sweep primitives.

All lengths are in mm. A pass is a straight sweep of the sensor along `sweep`, centered on the
point `center + standoff * direction`, with the sensor looking along `view = -direction`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def fibonacci_cap(n: int, min_elevation_deg: float) -> np.ndarray:
    """`n` unit vectors spread area-uniformly over the cap with elevation >= `min_elevation_deg`."""
    z0 = np.sin(np.radians(min_elevation_deg))
    i = np.arange(n)
    z = z0 + (1.0 - z0) * (i + 0.5) / n
    r = np.sqrt(1.0 - z * z)
    phi = i * GOLDEN_ANGLE
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def sweep_directions(view: np.ndarray) -> np.ndarray:
    """Global x projected onto the plane normal to each view direction.

    Defined everywhere on the cap (including the zenith): |x - (x.d) d| >= sin(min elevation).
    """
    x = np.array([1.0, 0.0, 0.0])
    s = x - (view @ x)[:, None] * view
    return s / np.linalg.norm(s, axis=1, keepdims=True)


def sensor_rotations(view: np.ndarray, sweep: np.ndarray) -> np.ndarray:
    """(N, 3, 3) sensor orientations: columns x = sweep, y = view x sweep (laser line), z = view."""
    return np.stack([sweep, np.cross(view, sweep), view], axis=2)


@dataclass(frozen=True)
class ScanPasses:
    center: np.ndarray     # (3,) part bbox center
    direction: np.ndarray  # (N, 3) unit, from center towards the sensor
    view: np.ndarray       # (N, 3) unit, = -direction
    sweep: np.ndarray      # (N, 3) unit, perpendicular to view
    standoff: np.ndarray   # (N,) distance from center to the sweep midpoint
    length: np.ndarray     # (N,) sweep length

    @property
    def n(self) -> int:
        return len(self.direction)

    @property
    def midpoint(self) -> np.ndarray:
        return self.center + self.standoff[:, None] * self.direction

    def samples(self, k: int) -> np.ndarray:
        """(N, k, 3) sensor positions from start (t = -L/2) to end (t = +L/2)."""
        t = np.linspace(-0.5, 0.5, k)[None, :, None] * self.length[:, None, None]
        return self.midpoint[:, None, :] + t * self.sweep[:, None, :]

    @property
    def rotation(self) -> np.ndarray:
        return sensor_rotations(self.view, self.sweep)

    @classmethod
    def from_cache(cls, z) -> ScanPasses:
        """Rebuild from a robot-plan cache file (scripts/reachability.py)."""
        return cls(*(np.asarray(z[f]) for f in ("center", "direction", "view", "sweep", "standoff", "length")))


def make_passes(
    vertices: np.ndarray,
    n: int,
    min_elevation_deg: float,
    standoff_offset_mm: float,
    margin_mm: float,
) -> ScanPasses:
    """Scan passes for one part along `n` Fibonacci cap directions (see `passes_for_directions`)."""
    return passes_for_directions(vertices, fibonacci_cap(n, min_elevation_deg), standoff_offset_mm, margin_mm)


def passes_for_directions(
    vertices: np.ndarray, directions: np.ndarray, standoff_offset_mm: float, margin_mm: float
) -> ScanPasses:
    """Scan passes for one part along given unit `directions` (center -> sensor).

    Standoff R(u) = h(u) + `standoff_offset_mm`, with h(u) the part's support in direction u, so the
    near limit of the depth of field touches the part's outermost point (D-020). The sweep is centered
    on the midpoint and covers the part's projection onto the sweep axis plus `margin_mm` on each side.
    """
    lo, hi = vertices.min(0), vertices.max(0)
    center = (lo + hi) / 2
    hull = vertices[ConvexHull(vertices).vertices] - center
    u = np.asarray(directions, dtype=float)
    view = -u
    s = sweep_directions(view)
    h = (hull @ u.T).max(0)
    proj = hull @ s.T
    return ScanPasses(
        center=center,
        direction=u,
        view=view,
        sweep=s,
        standoff=h + standoff_offset_mm,
        length=2 * np.abs(proj).max(0) + 2 * margin_mm,  # sweep is centered on the midpoint
    )


def standoff_offset(sensor: dict) -> float:
    """Working distance minus half the depth of field (mm)."""
    return sensor["working_distance_mm"] - sensor["depth_of_field_mm"] / 2
