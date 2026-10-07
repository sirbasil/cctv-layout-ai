"""Candidate camera positions derived from the CanonicalLayout masks.

Cameras are mounted on walls or ceiling perimeters, so candidates live on a wall-adjacent band:
interior pixels inset a safe, fixed distance (``wall_offset_m``) into valid interior space away from
the nearest wall or building edge. On that band we place:

  * corner candidates  - corners of the interior free space (room corners, wall intersections),
                         detected with corner features and inset onto the band;
  * wall candidates    - evenly spaced samples along the band (covering every wall and perimeter).

Every candidate is guaranteed to be strictly inside ``interior_free_mask``, inset into interior space,
and never inside a wall or exterior space. The input masks are only read, never modified.
"""
from dataclasses import dataclass
from typing import List, Literal

import cv2
import numpy as np

from ..models.layout import CanonicalLayout


@dataclass(frozen=True)
class Candidate:
    """A possible camera mount point in normalized-image pixel coordinates."""

    x: int
    y: int
    kind: Literal["corner", "wall"]


def _px(meters: float, meters_per_pixel: float, lo: int, hi: int) -> int:
    return int(np.clip(round(meters / meters_per_pixel), lo, hi))


def generate_candidates(
    layout: CanonicalLayout,
    meters_per_pixel: float,
    spacing_m: float = 1.0,
    wall_offset_m: float = 0.20,
    max_corners: int = 400,
) -> List[Candidate]:
    """Return wall-adjacent candidate positions inset into valid interior space.

    Args:
        layout: preprocessed floor plan.
        meters_per_pixel: scale of the normalized image.
        spacing_m: approximate distance between neighbouring wall candidates.
        wall_offset_m: distance (meters) by which cameras are inset into interior space
            from the nearest wall / building boundary. Default 0.20m (~8 px).
        max_corners: cap on detected corners.
    """
    if meters_per_pixel <= 0:
        raise ValueError("meters_per_pixel must be > 0")
    interior_u8 = layout.interior_free_mask
    interior = interior_u8 > 0
    if not interior.any():
        return []

    offset_px = _px(wall_offset_m, meters_per_pixel, 2, 50)
    spacing_px = _px(spacing_m, meters_per_pixel, 4, 10_000)

    # Distance from each interior pixel to the nearest non-interior pixel (wall or exterior).
    dist = cv2.distanceTransform(interior_u8, cv2.DIST_L2, 5)

    # Select interior pixels inset at the target distance from walls / perimeter
    band = interior & (dist >= offset_px) & (dist < offset_px + 1.5)
    if not band.any():  # narrow spaces / small scale: fall back to deepest interior line
        band = interior & (dist >= max(1.0, min(float(offset_px), float(dist.max())) - 0.5))
    band &= layout.wall_mask == 0  # guaranteed by contract, but explicit

    by, bx = np.nonzero(band)
    if bx.size == 0:
        return []
    band_pts = np.stack([bx, by], axis=1).astype(np.float32)

    kept: List[Candidate] = []
    kept_xy: List[tuple] = []
    min_sep2 = (0.5 * spacing_px) ** 2

    def _try_add(x: int, y: int, kind: str, sep2: float) -> None:
        for kx, ky in kept_xy:
            if (kx - x) ** 2 + (ky - y) ** 2 < sep2:
                return
        kept.append(Candidate(int(x), int(y), kind))  # type: ignore[arg-type]
        kept_xy.append((x, y))

    # 1) corners / wall intersections of the interior space, inset onto the wall band
    corners = cv2.goodFeaturesToTrack(
        interior_u8, maxCorners=max_corners, qualityLevel=0.05,
        minDistance=max(3.0, 0.5 * spacing_px), blockSize=5,
    )
    snap_r2 = (2.5 * offset_px + 3) ** 2
    if corners is not None:
        for cx, cy in corners.reshape(-1, 2):
            d2 = (band_pts[:, 0] - cx) ** 2 + (band_pts[:, 1] - cy) ** 2
            j = int(np.argmin(d2))
            if d2[j] <= snap_r2:
                _try_add(int(band_pts[j, 0]), int(band_pts[j, 1]), "corner", (0.25 * spacing_px) ** 2)

    # 2) evenly spaced wall / perimeter points: one band pixel per grid cell (closest to centre)
    cell_x, cell_y = bx // spacing_px, by // spacing_px
    centre_d2 = ((bx % spacing_px) - spacing_px / 2) ** 2 + ((by % spacing_px) - spacing_px / 2) ** 2
    cell_id = cell_y.astype(np.int64) * (layout.width // spacing_px + 2) + cell_x
    order = np.lexsort((centre_d2, cell_id))
    _, first = np.unique(cell_id[order], return_index=True)
    for j in order[first]:
        _try_add(int(bx[j]), int(by[j]), "wall", min_sep2)

    return kept
