"""Coverage metrics and blind-spot extraction.

Only ``interior_free_mask`` counts: walls and the building exterior are neither "covered" nor
"blind". Coverage is evaluated at full pixel resolution from the union of per-camera visibility
polygons (see ``visibility.camera_coverage_mask``).
"""
from dataclasses import dataclass, field
from typing import List, Literal, Sequence, Tuple

import cv2
import numpy as np

from ..models.layout import CanonicalLayout
from .visibility import CameraModel, camera_coverage_mask, wall_distance_map

PriorityClass = Literal["high", "medium", "low"]
SizeClass = Literal["large", "medium", "small"]


def classify_blind_spot(
    area_m2: float,
    large_threshold_m2: float = 4.0,
    medium_threshold_m2: float = 1.0,
) -> Tuple[PriorityClass, SizeClass]:
    """Classify a blind spot by area into priority and size classes.

    Args:
        area_m2: Area of the blind spot in square meters.
        large_threshold_m2: Areas at or above this are high priority / large.
        medium_threshold_m2: Areas at or above this (and below large) are medium priority / medium.
    """
    if area_m2 >= large_threshold_m2:
        return "high", "large"
    elif area_m2 >= medium_threshold_m2:
        return "medium", "medium"
    return "low", "small"


@dataclass(frozen=True)
class BlindSpot:
    """One connected uncovered interior region with area, centroid, bounding box, and priority."""

    centroid: Tuple[float, float]  # (x, y) px
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) px
    area_px: int
    area_m2: float
    priority: PriorityClass  # "high", "medium", "low"
    size_class: SizeClass  # "large", "medium", "small"
    meters_per_pixel: float = 0.0

    @property
    def centroid_m(self) -> Tuple[float, float]:
        """Centroid coordinates in real-world meters (x, y)."""
        return (self.centroid[0] * self.meters_per_pixel, self.centroid[1] * self.meters_per_pixel)

    @property
    def bbox_m(self) -> Tuple[float, float, float, float]:
        """Bounding box in real-world meters (x, y, w, h)."""
        return tuple(v * self.meters_per_pixel for v in self.bbox)

    def to_dict(self) -> dict:
        return {
            "area_m2": round(self.area_m2, 3),
            "area_px": self.area_px,
            "centroid": [round(c, 1) for c in self.centroid],
            "centroid_m": [round(c, 2) for c in self.centroid_m] if self.meters_per_pixel > 0 else [],
            "bbox": list(self.bbox),
            "bbox_m": [round(v, 2) for v in self.bbox_m] if self.meters_per_pixel > 0 else [],
            "priority": self.priority,
            "size_class": self.size_class,
        }


@dataclass
class CoverageReport:
    total_area_px: int
    covered_area_px: int
    meters_per_pixel: float
    covered_mask: np.ndarray = field(repr=False)  # bool (H, W), subset of interior
    blind_mask: np.ndarray = field(repr=False)  # bool (H, W), interior AND NOT covered
    blind_spots: List[BlindSpot] = field(default_factory=list)  # regions >= min area

    @property
    def blind_area_px(self) -> int:
        return self.total_area_px - self.covered_area_px

    @property
    def coverage_percent(self) -> float:
        if self.total_area_px == 0:
            return 0.0
        return 100.0 * self.covered_area_px / self.total_area_px

    @property
    def total_area_m2(self) -> float:
        return self.total_area_px * self.meters_per_pixel ** 2

    @property
    def covered_area_m2(self) -> float:
        return self.covered_area_px * self.meters_per_pixel ** 2

    @property
    def blind_area_m2(self) -> float:
        return self.blind_area_px * self.meters_per_pixel ** 2

    @property
    def blind_spot_counts_by_priority(self) -> dict:
        return {
            "high": sum(1 for b in self.blind_spots if b.priority == "high"),
            "medium": sum(1 for b in self.blind_spots if b.priority == "medium"),
            "low": sum(1 for b in self.blind_spots if b.priority == "low"),
        }

    @property
    def blind_spot_counts_by_size(self) -> dict:
        return {
            "large": sum(1 for b in self.blind_spots if b.size_class == "large"),
            "medium": sum(1 for b in self.blind_spots if b.size_class == "medium"),
            "small": sum(1 for b in self.blind_spots if b.size_class == "small"),
        }

    def to_dict(self) -> dict:
        return {
            "total_area_m2": round(self.total_area_m2, 2),
            "covered_area_m2": round(self.covered_area_m2, 2),
            "blind_area_m2": round(self.blind_area_m2, 2),
            "coverage_percent": round(self.coverage_percent, 2),
            "blind_spot_count": len(self.blind_spots),
            "blind_spot_counts_by_priority": self.blind_spot_counts_by_priority,
            "blind_spot_counts_by_size": self.blind_spot_counts_by_size,
            "blind_spots": [b.to_dict() for b in self.blind_spots],
        }


def find_blind_spots(
    blind_mask: np.ndarray,
    meters_per_pixel: float,
    min_area_m2: float = 0.25,
    large_threshold_m2: float = 4.0,
    medium_threshold_m2: float = 1.0,
) -> List[BlindSpot]:
    """Connected uncovered regions with area >= min_area_m2, largest first.

    Each returned BlindSpot includes area, centroid, bounding box, priority, and size class.
    Tiny slivers (anti-aliasing / rasterization noise along walls) are still counted in the
    blind AREA, they are just not reported as separate blind spots.
    """
    n, _, stats, cents = cv2.connectedComponentsWithStats(blind_mask.astype(np.uint8), connectivity=8)
    min_px = max(1, int(round(min_area_m2 / meters_per_pixel ** 2)))
    spots = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_px:
            continue
        x, y, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                 cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        area_m2 = area * meters_per_pixel ** 2
        priority, size_class = classify_blind_spot(
            area_m2,
            large_threshold_m2=large_threshold_m2,
            medium_threshold_m2=medium_threshold_m2,
        )
        spots.append(
            BlindSpot(
                centroid=(float(cents[i, 0]), float(cents[i, 1])),
                bbox=(x, y, w, h),
                area_px=area,
                area_m2=area_m2,
                priority=priority,
                size_class=size_class,
                meters_per_pixel=meters_per_pixel,
            )
        )
    spots.sort(key=lambda b: b.area_px, reverse=True)
    return spots


def report_from_mask(
    layout: CanonicalLayout,
    covered: np.ndarray,
    meters_per_pixel: float,
    min_blind_area_m2: float = 0.25,
) -> CoverageReport:
    interior = layout.interior_free_mask > 0
    covered = np.asarray(covered, bool) & interior
    blind = interior & ~covered
    return CoverageReport(
        total_area_px=int(interior.sum()),
        covered_area_px=int(covered.sum()),
        meters_per_pixel=meters_per_pixel,
        covered_mask=covered,
        blind_mask=blind,
        blind_spots=find_blind_spots(blind, meters_per_pixel, min_blind_area_m2),
    )


def compute_coverage(
    layout: CanonicalLayout,
    cameras: Sequence[Tuple[float, float, float]],
    camera: CameraModel,
    meters_per_pixel: float,
    min_blind_area_m2: float = 0.25,
    dist_map: np.ndarray = None,
) -> CoverageReport:
    """Coverage of a set of cameras given as (x, y, orientation_deg) tuples."""
    if dist_map is None:
        dist_map = wall_distance_map(layout.wall_mask)
    covered = np.zeros((layout.height, layout.width), bool)
    for x, y, orient in cameras:
        m, _ = camera_coverage_mask((x, y), orient, camera, meters_per_pixel,
                                    layout.interior_free_mask, dist_map)
        covered |= m
    return report_from_mask(layout, covered, meters_per_pixel, min_blind_area_m2)


def coverage_mask_image(layout: CanonicalLayout, report: CoverageReport) -> np.ndarray:
    """uint8 (H, W): 255 = covered interior, 100 = blind interior, 0 = wall / exterior."""
    img = np.zeros((layout.height, layout.width), np.uint8)
    img[report.blind_mask] = 100
    img[report.covered_mask] = 255
    return img


def blind_spot_mask_image(layout: CanonicalLayout, report: CoverageReport) -> np.ndarray:
    """uint8 (H, W): 255 = blind interior (uncovered area), 0 = covered interior / exterior / wall."""
    img = np.zeros((layout.height, layout.width), np.uint8)
    img[report.blind_mask] = 255
    return img
