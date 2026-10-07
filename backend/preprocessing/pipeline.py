"""One entry point: file path in, CanonicalLayout out.

    file -> load -> normalize -> wall mask -> raw free space
         -> building footprint -> interior free mask -> CanonicalLayout
"""
import warnings
from typing import Optional

from ..models.layout import CanonicalLayout
from .footprint import build_building_mask, get_interior_free_mask
from .loader import get_source_format, load_layout
from .normalize import normalize_image
from .walls import extract_wall_mask, get_free_space


def preprocess(
    path,
    max_dimension: int = 1200,
    meters_per_pixel: Optional[float] = None,
    min_wall_thickness: Optional[int] = None,
    footprint_gap_ratio: float = 0.10,
) -> CanonicalLayout:
    """Run the full preprocessing pipeline on a PNG / JPG / JPEG / PDF floor plan.

    Args:
        path: input file.
        max_dimension: longest side (px) of the normalized working image.
        meters_per_pixel: optional real-world scale of the NORMALIZED image
            (not the original file, which may have been resized).
        min_wall_thickness: passed to extract_wall_mask. None = estimate automatically
            (default); 0/1 = disable; n >= 2 = explicit threshold in pixels.
        footprint_gap_ratio: widest outer-wall opening (as a fraction of the longer image
            side) that is bridged when locating the building footprint. The final
            wall_mask is NOT affected by this.
    """
    source_format = get_source_format(path)  # fail fast on unsupported types

    raw = load_layout(path)
    image = normalize_image(raw, max_dimension=max_dimension)
    wall_mask = extract_wall_mask(image, min_wall_thickness=min_wall_thickness)
    free_space_mask = get_free_space(wall_mask)  # raw inverse, includes the exterior
    building_mask = build_building_mask(wall_mask, max_gap_ratio=footprint_gap_ratio)
    interior_free_mask = get_interior_free_mask(building_mask, wall_mask)

    if not interior_free_mask.any():
        warnings.warn(
            f"No enclosed interior found in '{path}': the walls may be missing, or the outer "
            "boundary has an opening wider than footprint_gap_ratio. "
            "interior_free_mask is empty.",
            RuntimeWarning,
            stacklevel=2,
        )

    height, width = image.shape[:2]
    return CanonicalLayout(
        source_format=source_format,
        source_path=str(path),
        width=width,
        height=height,
        meters_per_pixel=meters_per_pixel,
        wall_mask=wall_mask,
        free_space_mask=free_space_mask,
        building_mask=building_mask,
        interior_free_mask=interior_free_mask,
        image=image,
    )
