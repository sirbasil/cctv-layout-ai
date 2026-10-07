"""CCTV analysis on top of a CanonicalLayout.

    CanonicalLayout -> candidates -> visibility -> coverage / blind spots -> greedy optimizer
"""
from .candidates import Candidate, generate_candidates
from .coverage import (BlindSpot, CoverageReport, blind_spot_mask_image, classify_blind_spot,
                       compute_coverage, coverage_mask_image, find_blind_spots)
from .optimizer import OptimizationResult, PlacedCamera, meters_per_pixel_from_width, optimize_layout
from .visibility import (CameraModel, march_rays, sample_interior_points, visibility_polygon,
                         visible_points, wall_distance_map)
from .visualize import render_layout, save_outputs

__all__ = [
    "BlindSpot",
    "CameraModel",
    "Candidate",
    "CoverageReport",
    "OptimizationResult",
    "PlacedCamera",
    "blind_spot_mask_image",
    "classify_blind_spot",
    "compute_coverage",
    "coverage_mask_image",
    "find_blind_spots",
    "generate_candidates",
    "march_rays",
    "meters_per_pixel_from_width",
    "optimize_layout",
    "render_layout",
    "sample_interior_points",
    "save_outputs",
    "visibility_polygon",
    "visible_points",
    "wall_distance_map",
]
