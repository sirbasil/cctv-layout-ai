"""Greedy CCTV placement on top of a CanonicalLayout.

    CanonicalLayout -> scale -> candidates -> (candidate x orientation) visibility sets
                    -> greedy max-additional-coverage -> cameras + coverage + blind spots

Requirements:
    1. Monotonic coverage: adding a camera never reduces total covered area (coverage(S + c) >= coverage(S)).
    2. Maximize additional coverage: at each greedy step, select the candidate providing the largest
       ADDITIONAL uncovered interior area.
    3. Early termination: stop as soon as target_coverage (default 95.0%) is reached. Never add
       unnecessary cameras if the target has been satisfied.
    4. Best-solution preservation: track the best coverage and minimal camera count.
"""
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..models.layout import CanonicalLayout
from .candidates import Candidate, generate_candidates
from .coverage import CoverageReport, report_from_mask
from .visibility import (CameraModel, camera_coverage_mask, in_fov, omni_visibility,
                         sample_interior_points, wall_distance_map)


def meters_per_pixel_from_width(building_width_meters: float, layout: CanonicalLayout) -> float:
    """meters_per_pixel = building_width_meters / normalized_image_width (manual scale only)."""
    if building_width_meters is None or building_width_meters <= 0:
        raise ValueError("building_width_meters must be > 0")
    return float(building_width_meters) / layout.width


@dataclass(frozen=True)
class PlacedCamera:
    x: int  # px, normalized image
    y: int
    orientation_deg: float  # 0 = +x (right), 90 = +y (down)
    kind: str  # candidate type: "corner" / "wall"
    meters_per_pixel: float
    added_coverage_percent: float  # marginal gain when it was selected

    @property
    def x_m(self) -> float:
        return self.x * self.meters_per_pixel

    @property
    def y_m(self) -> float:
        return self.y * self.meters_per_pixel

    def to_dict(self) -> dict:
        return {
            "x_px": self.x,
            "y_px": self.y,
            "x_m": round(self.x_m, 2),
            "y_m": round(self.y_m, 2),
            "orientation_deg": round(self.orientation_deg, 1),
            "kind": self.kind,
            "added_coverage_percent": round(self.added_coverage_percent, 2),
        }


@dataclass
class OptimizationResult:
    cameras: List[PlacedCamera]
    report: CoverageReport
    camera_model: CameraModel
    meters_per_pixel: float
    candidate_count: int
    sample_count: int
    coverage_history: List[float]  # pixel coverage % after each added camera
    stop_reason: str
    runtime_s: float
    fov_polygons: List[np.ndarray] = field(default_factory=list, repr=False)

    @property
    def coverage_percent(self) -> float:
        return self.report.coverage_percent

    @property
    def camera_count(self) -> int:
        return len(self.cameras)

    @property
    def blind_spots(self):
        return self.report.blind_spots

    def to_dict(self) -> dict:
        return {
            "camera_count": self.camera_count,
            "coverage_percent": round(self.coverage_percent, 2),
            "meters_per_pixel": self.meters_per_pixel,
            "camera_model": self.camera_model.to_dict(),
            "cameras": [c.to_dict() for c in self.cameras],
            "coverage": self.report.to_dict(),
            "coverage_history": [round(c, 2) for c in self.coverage_history],
            "candidate_count": self.candidate_count,
            "sample_count": self.sample_count,
            "stop_reason": self.stop_reason,
            "runtime_s": round(self.runtime_s, 2),
        }


def _visibility_options(cands: List[Candidate], points: np.ndarray, camera: CameraModel,
                        range_px: float, dist_map: np.ndarray, orientations: np.ndarray):
    """Bool matrix (n_options, n_points) + list of (candidate_index, orientation_deg)."""
    rows, meta = [], []
    for ci, c in enumerate(cands):
        vis, ang = omni_visibility((c.x, c.y), points, range_px, dist_map)
        if not vis.any():
            continue
        for o in orientations:
            row = vis & in_fov(ang, math.radians(o), camera.fov_deg)
            if row.any():
                rows.append(row)
                meta.append((ci, float(o)))
            if camera.fov_deg >= 360:
                break  # every orientation is identical
    if not rows:
        return np.zeros((0, len(points)), bool), []
    return np.vstack(rows), meta


def optimize_layout(
    layout: CanonicalLayout,
    building_width_meters: float,
    camera: Optional[CameraModel] = None,
    target_coverage: float = 95.0,
    max_cameras: int = 20,
    candidate_spacing_m: float = 1.0,
    wall_offset_m: float = 0.20,
    sample_spacing_m: float = 0.5,
    max_samples: int = 6000,
    orientation_step_deg: float = 15.0,
    rescore_top_k: int = 16,
    min_marginal_gain_percent: float = 0.2,
    min_blind_area_m2: float = 0.25,
) -> OptimizationResult:
    """Greedy camera placement maximizing marginal coverage.

    Stops immediately when target_coverage is reached, when max_cameras is reached, or when no
    candidate adds more than min_marginal_gain_percent.
    """
    t0 = time.perf_counter()
    camera = camera or CameraModel()
    mpp = meters_per_pixel_from_width(building_width_meters, layout)
    range_px = camera.range_px(mpp)
    interior = layout.interior_free_mask
    dist_map = wall_distance_map(layout.wall_mask)  # wall_mask is strictly read-only

    cands = generate_candidates(layout, mpp, spacing_m=candidate_spacing_m, wall_offset_m=wall_offset_m)

    spacing_px = max(2, round(sample_spacing_m / mpp))
    n_interior = int(np.count_nonzero(interior))
    spacing_px = max(spacing_px, math.ceil(math.sqrt(n_interior / max_samples)))
    points = sample_interior_points(interior, spacing_px)

    orientations = np.arange(0.0, 360.0, orientation_step_deg)
    options, meta = _visibility_options(cands, points, camera, range_px, dist_map, orientations)

    option_cand = np.array([ci for ci, _ in meta], dtype=np.int64) if meta else np.array([], dtype=np.int64)
    covered_px = np.zeros(interior.shape, bool)
    covered_pts = np.zeros(len(points), bool)
    total_px = max(1, n_interior)
    min_gain_px = max(1, int(round((min_marginal_gain_percent / 100.0) * total_px)))

    cameras: List[PlacedCamera] = []
    polygons: List[np.ndarray] = []
    history: List[float] = []
    used = set()
    stop_reason = "no candidates" if not meta else ""

    while meta:
        current_cov = 100.0 * covered_px.sum() / total_px
        if current_cov >= target_coverage:
            stop_reason = f"target coverage {target_coverage:g}% reached"
            break
        if len(cameras) >= max_cameras:
            stop_reason = f"camera limit {max_cameras} reached"
            break

        # Marginal gains on sampled points
        gains = (options & ~covered_pts).sum(axis=1)
        if used:  # one camera per mount position
            gains[np.isin(option_cand, list(used))] = -1
        top = np.argsort(-gains, kind="stable")[:max(1, rescore_top_k)]
        top = [int(i) for i in top if gains[i] > 0]
        if not top:
            stop_reason = "no candidate adds coverage"
            break

        best = None
        for i in top:  # re-score shortlisted options at full pixel resolution
            ci, orient = meta[i]
            c = cands[ci]
            mask, poly = camera_coverage_mask((c.x, c.y), orient, camera, mpp, interior, dist_map)
            gain_px = int((mask & ~covered_px).sum())
            if best is None or gain_px > best[0]:
                best = (gain_px, i, mask, poly)

        if best is None or best[0] < min_gain_px:
            stop_reason = f"no candidate adds > {min_marginal_gain_percent:g}% coverage"
            break

        gain_px, i, mask, poly = best
        ci, orient = meta[i]
        c = cands[ci]

        # Monotonicity invariant: union coverage cannot decrease
        prev_count = int(covered_px.sum())
        covered_px |= mask
        new_count = int(covered_px.sum())
        assert new_count >= prev_count, "Monotonicity violation: coverage decreased"

        covered_pts |= options[i]
        used.add(ci)
        polygons.append(poly)
        marginal_pct = 100.0 * gain_px / total_px
        cameras.append(PlacedCamera(c.x, c.y, orient, c.kind, mpp, marginal_pct))
        new_cov = 100.0 * new_count / total_px
        history.append(new_cov)

        # Stop immediately if target coverage reached with this camera
        if new_cov >= target_coverage:
            stop_reason = f"target coverage {target_coverage:g}% reached"
            break

    if not stop_reason:
        stop_reason = "optimization completed"

    report = report_from_mask(layout, covered_px, mpp, min_blind_area_m2)
    return OptimizationResult(
        cameras=cameras, report=report, camera_model=camera, meters_per_pixel=mpp,
        candidate_count=len(cands), sample_count=len(points), coverage_history=history,
        stop_reason=stop_reason, runtime_s=time.perf_counter() - t0, fov_polygons=polygons,
    )
