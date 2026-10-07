"""Camera model + line-of-sight visibility on the CanonicalLayout masks.

Coordinate / angle convention (normalized-image pixels):
    x grows to the right, y grows DOWN (image rows).
    orientation_deg = 0 points right (+x), 90 points down (+y), i.e. clockwise on screen.

A target point is visible from a camera only if
    * it is inside interior_free_mask        (caller passes interior points only)
    * distance <= camera range
    * it lies inside the camera's horizontal FOV wedge
    * the straight segment camera -> point never touches a wall_mask pixel

Line of sight uses sphere tracing on a distance transform of the non-wall space: from any
position we can safely jump (distance-to-nearest-wall - 1.5 px) without crossing a wall pixel,
and fall back to 0.5 px steps close to walls. This is exact up to pixel discretization and fast
on CPU. wall_mask is only read, never modified.

This is a simple 2D geometric model for a hackathon prototype - no lens distortion, mounting
height, resolution / pixels-per-face density, lighting or vertical FOV.
"""
from dataclasses import asdict, dataclass
from typing import Tuple

import cv2
import numpy as np

_MIN_STEP = 0.5  # px, step used right next to walls
_SAFETY = 1.5  # px, margin so a jump can never clip the corner of a wall pixel


@dataclass(frozen=True)
class CameraModel:
    """Configurable, idealised camera."""

    fov_deg: float = 90.0
    max_range_m: float = 15.0

    def __post_init__(self):
        if not 0 < self.fov_deg <= 360:
            raise ValueError("fov_deg must be in (0, 360]")
        if self.max_range_m <= 0:
            raise ValueError("max_range_m must be > 0")

    def range_px(self, meters_per_pixel: float) -> float:
        return self.max_range_m / meters_per_pixel

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------- primitives
def wall_distance_map(wall_mask: np.ndarray) -> np.ndarray:
    """float32 map: distance (px) from every pixel to the nearest wall pixel (0 on walls)."""
    free = (wall_mask == 0).astype(np.uint8)
    return cv2.distanceTransform(free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def march_rays(
    origin: Tuple[float, float],
    targets: np.ndarray,
    dist_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trace straight segments from ``origin`` towards each target (N, 2) [x, y].

    Returns:
        reached: bool (N,) - True if the segment reaches the target without touching a wall
                 (or leaving the image).
        travel:  float (N,) - distance travelled before stopping (= full length if reached).
    """
    h, w = dist_map.shape
    o = np.asarray(origin, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
    vec = t - o
    total = np.hypot(vec[:, 0], vec[:, 1])
    reached = np.zeros(len(t), bool)
    travel = np.zeros(len(t))

    trivial = total < 1e-9
    reached[trivial] = True
    idx = np.nonzero(~trivial)[0]
    if idx.size == 0:
        return reached, travel

    dirs = vec[idx] / total[idx, None]
    pos = np.repeat(o[None, :], idx.size, axis=0)
    remaining = total[idx].copy()
    trav = np.zeros(idx.size)

    max_iter = int(total.max() / _MIN_STEP) + 4
    for _ in range(max_iter):
        xi = np.rint(pos[:, 0]).astype(np.intp)
        yi = np.rint(pos[:, 1]).astype(np.intp)
        inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        d = np.zeros(idx.size, np.float32)
        d[inside] = dist_map[yi[inside], xi[inside]]

        blocked = d <= 0  # wall pixel or outside the image
        step = np.maximum(d - _SAFETY, _MIN_STEP)
        done = ~blocked & (step >= remaining)

        reached[idx[done]] = True
        travel[idx[done]] = total[idx[done]]
        travel[idx[blocked]] = trav[blocked]

        keep = ~(blocked | done)
        if not keep.any():
            break
        idx, dirs, pos = idx[keep], dirs[keep], pos[keep]
        remaining, trav, step = remaining[keep], trav[keep], step[keep]
        pos += dirs * step[:, None]
        remaining -= step
        trav += step
    return reached, travel


def angle_diff(a: np.ndarray, b: float) -> np.ndarray:
    """Smallest absolute difference between angles (radians), in [0, pi]."""
    return np.abs((np.asarray(a) - b + np.pi) % (2 * np.pi) - np.pi)


def sample_interior_points(interior_mask: np.ndarray, spacing_px: int) -> np.ndarray:
    """Regular grid of interior points (N, 2) [x, y]; every point lies in interior_free_mask."""
    spacing_px = max(1, int(spacing_px))
    h, w = interior_mask.shape
    off = spacing_px // 2
    ys, xs = np.mgrid[off:h:spacing_px, off:w:spacing_px]
    keep = interior_mask[ys, xs] > 0
    return np.stack([xs[keep], ys[keep]], axis=1).astype(np.int32)


# ------------------------------------------------------------------------------- visibility
def omni_visibility(
    origin: Tuple[float, float],
    points: np.ndarray,
    range_px: float,
    dist_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """360 deg visibility (range + line of sight, no FOV yet) from origin to points.

    Returns (visible bool (N,), angle_rad (N,)). Combine with ``in_fov`` per orientation.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    dx, dy = pts[:, 0] - origin[0], pts[:, 1] - origin[1]
    dist = np.hypot(dx, dy)
    angles = np.arctan2(dy, dx)
    visible = np.zeros(len(pts), bool)
    near = np.nonzero(dist <= range_px)[0]
    if near.size:
        visible[near], _ = march_rays(origin, pts[near], dist_map)
    return visible, angles


def in_fov(angles: np.ndarray, orientation_rad: float, fov_deg: float) -> np.ndarray:
    if fov_deg >= 360:
        return np.ones(np.shape(angles), bool)
    return angle_diff(angles, orientation_rad) <= np.radians(fov_deg) / 2 + 1e-9


def visible_points(
    origin: Tuple[float, float],
    orientation_deg: float,
    points: np.ndarray,
    camera: CameraModel,
    meters_per_pixel: float,
    wall_mask: np.ndarray,
    interior_mask: np.ndarray,
    dist_map: np.ndarray = None,
) -> np.ndarray:
    """Full visibility test for arbitrary points: interior + range + FOV + line of sight."""
    if dist_map is None:
        dist_map = wall_distance_map(wall_mask)
    pts = np.asarray(points).reshape(-1, 2)
    xi = np.clip(np.rint(pts[:, 0]).astype(int), 0, interior_mask.shape[1] - 1)
    yi = np.clip(np.rint(pts[:, 1]).astype(int), 0, interior_mask.shape[0] - 1)
    interior = interior_mask[yi, xi] > 0
    vis, ang = omni_visibility(origin, pts, camera.range_px(meters_per_pixel), dist_map)
    return interior & vis & in_fov(ang, np.radians(orientation_deg), camera.fov_deg)


def visibility_polygon(
    origin: Tuple[float, float],
    orientation_deg: float,
    camera: CameraModel,
    meters_per_pixel: float,
    dist_map: np.ndarray,
    ray_spacing_px: float = 0.75,
) -> np.ndarray:
    """Approximate visible region of one camera as a polygon (K, 2) int32 [x, y].

    Rays are cast across the FOV, densely enough that neighbouring rays are at most
    ``ray_spacing_px`` apart at full range; each stops at the first wall pixel.
    """
    range_px = camera.range_px(meters_per_pixel)
    fov = np.radians(camera.fov_deg)
    n = max(16, int(np.ceil(fov * range_px / ray_spacing_px)) + 1)
    o = np.radians(orientation_deg)
    full = camera.fov_deg >= 360
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) if full else np.linspace(o - fov / 2, o + fov / 2, n)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    targets = np.asarray(origin, float)[None, :] + range_px * dirs
    _, travel = march_rays(origin, targets, dist_map)
    pts = np.asarray(origin, float)[None, :] + travel[:, None] * dirs
    if not full:
        pts = np.vstack([np.asarray(origin, float)[None, :], pts])
    return np.rint(pts).astype(np.int32)


def camera_coverage_mask(
    origin: Tuple[float, float],
    orientation_deg: float,
    camera: CameraModel,
    meters_per_pixel: float,
    interior_mask: np.ndarray,
    dist_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel-level coverage of one camera: (bool mask restricted to interior, polygon)."""
    poly = visibility_polygon(origin, orientation_deg, camera, meters_per_pixel, dist_map)
    mask = np.zeros(interior_mask.shape, np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return (mask > 0) & (interior_mask > 0), poly
