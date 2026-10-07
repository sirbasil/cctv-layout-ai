"""Wall extraction for clean 2D architectural floor plans (classical OpenCV, no ML).

CONTRACT
    wall_mask: uint8 (H, W), 255 = structural wall / visibility obstacle, 0 = not a wall.
    Door openings are NOT closed here: a door is a real gap in the wall and the CCTV
    visibility stage needs that gap. (Footprint detection, which does need to bridge
    doors, works on a temporary copy - see footprint.py.)

PIPELINE
    grayscale -> Otsu threshold ("ink") -> thickness filter (drops thin lines)
              -> horizontal/vertical run extraction (drops text / arcs / curves)
              -> small gap closing -> connected-component filter (drops isolated bits)

WHAT IT RELIES ON
    Structural walls are drawn thicker AND longer than furniture outlines, dimension
    lines, text and hatching. The thickness threshold is estimated from the drawing
    itself, so it adapts to image size / drawing style instead of being a fixed pixel
    count.

LIMITATIONS (this is an approximation, NOT CAD-level geometry)
  * Works best on clean 2D architectural floor plans: dark, solid walls on a light
    background (a dark background is auto-detected and inverted).
  * Assumes mostly horizontal / vertical walls. Diagonal and curved walls are missed
    or reduced to fragments.
  * Thickness is the main cue separating walls from furniture. Plans whose walls are
    drawn as thin single lines, or as two thin parallel lines (hollow "CAD" walls),
    cannot be separated from furniture this way. If no wall is clearly thicker than a
    few pixels the thickness filter switches itself off, and furniture/dimension lines
    WILL then be detected. Hollow double-line walls are not filled.
  * Thin dimension lines are removed because they are thin, not because they are
    understood. A dimension line drawn as thick as a wall, a thick filled furniture
    block (bed, counter, column) or a title-block frame can still be detected as wall.
    Furniture may therefore still occasionally be detected - removal is not perfect.
  * Walls thinner than the estimated threshold (about a third of the typical wall
    thickness, at least 3 px) are lost, and so are free-standing wall stubs shorter than
    `min_component_ratio` of the image's shorter side.
  * Doors and windows are not classified. A door is just a gap in the wall (correct for
    visibility). A window drawn as a thin line inside a wall opens a gap in the mask;
    a window drawn as a filled glazing bar stays "wall".
  * Single global (Otsu) threshold: coloured/shaded rooms, gradients or heavy JPEG noise
    can pollute the mask.
  * Scanned documents, photos and skewed / rotated plans are outside the MVP target.
  * Free space from get_free_space() INCLUDES the exterior of the building. Use
    interior_free_mask (footprint.py) for anything that must stay inside the building.
"""
import warnings
from typing import Optional

import cv2
import numpy as np

# Structuring-element sizes must be ODD: OpenCV's even-sized kernels have an off-centre
# anchor, which makes morphological opening shift the result by one pixel.
_THICKNESS_FRACTION = 0.35  # filter strokes thinner than ~35% of the typical wall thickness
_MIN_ESTIMATE_PX = 5.0  # below this typical thickness there is nothing "thick" to separate
_MAX_KERNEL_RATIO = 0.006  # cap on the thickness kernel, as a fraction of the longer side


def extract_wall_mask(
    image: np.ndarray,
    min_line_length_ratio: float = 0.03,
    min_wall_thickness: Optional[int] = None,
    min_component_ratio: float = 0.05,
) -> np.ndarray:
    """Return a binary wall mask (uint8, wall = 255, non-wall = 0), same HxW as `image`.

    Args:
        image: BGR image, shape (H, W, 3).
        min_line_length_ratio: shortest straight H/V run kept as "wall", as a fraction
            of the image's shorter side. Smaller keeps shorter stubs but more clutter.
        min_wall_thickness: minimum stroke thickness (px) that counts as wall.
            None (default) = estimate automatically from the drawing (scale-aware).
            0 or 1 = disable the thickness filter (keeps thin-line walls AND furniture).
            n >= 2 = use n explicitly (rounded up to an odd size internally).
        min_component_ratio: connected wall pieces whose longest side is shorter than
            this fraction of the image's shorter side are discarded as isolated clutter.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("extract_wall_mask expects a BGR image of shape (H, W, 3)")

    h, w = image.shape[:2]
    ink = _binarize_ink(image)

    line_len = max(15, int(min(h, w) * min_line_length_ratio))
    min_extent = max(line_len, int(min(h, w) * min_component_ratio))

    auto = min_wall_thickness is None
    thickness = _auto_thickness(ink, max(h, w)) if auto else int(min_wall_thickness)

    walls = _filter_ink(ink, thickness, line_len, min_extent)

    if auto and thickness > 1 and ink.any() and not walls.any():
        # Safety net: the drawing is so thin that the thickness filter erased everything.
        warnings.warn(
            "extract_wall_mask: thickness filtering removed all ink; retrying without it. "
            "Furniture / dimension lines may be detected as walls.",
            RuntimeWarning,
            stacklevel=2,
        )
        walls = _filter_ink(ink, 1, line_len, min_extent)

    return walls


def get_free_space(wall_mask: np.ndarray) -> np.ndarray:
    """Return the RAW inverse of the wall mask: free = 255, wall = 0 (uint8).

    This includes the exterior of the building, so it must NOT be used to place cameras.
    Use the interior_free_mask from footprint.get_interior_free_mask() for that.
    """
    if not isinstance(wall_mask, np.ndarray) or wall_mask.ndim != 2:
        raise ValueError("get_free_space expects a 2D wall mask (H, W)")
    return np.where(wall_mask > 0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- helpers
def _binarize_ink(image: np.ndarray) -> np.ndarray:
    """Grayscale + Otsu threshold. Returns uint8 {0, 255} with drawn ink = 255."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # kills JPEG speckle without erasing lines
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # If most pixels are "ink", the plan is light-on-dark; flip so background = 0.
    if np.count_nonzero(ink) > 0.5 * ink.size:
        ink = cv2.bitwise_not(ink)
    return ink


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def _auto_thickness(ink: np.ndarray, longest_side: int) -> int:
    """Estimate a stroke-thickness threshold (px) from the drawing itself.

    The distance transform gives, for every ink pixel, the distance to the nearest
    background pixel (~ half the stroke thickness near a stroke's centre line). The 90th
    percentile over all ink pixels reflects the thickest common strokes, i.e. the walls.
    Returns 1 (= no thickness filtering) when nothing in the drawing is clearly thick.
    """
    if not ink.any():
        return 1
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    typical_thickness = 2.0 * float(np.percentile(dist[ink > 0], 90))
    if typical_thickness < _MIN_ESTIMATE_PX:
        return 1
    cap = max(3, int(round(_MAX_KERNEL_RATIO * longest_side)))
    return int(np.clip(round(_THICKNESS_FRACTION * typical_thickness), 3, cap))


def _filter_ink(ink: np.ndarray, thickness: int, line_len: int, min_extent: int) -> np.ndarray:
    """Thickness filter -> H/V run extraction -> gap closing -> component filter."""
    # 1. Drop strokes thinner than `thickness` (furniture, dimension lines, text, arcs).
    #    Opening with a k x k square leaves any shape that can contain a k x k block
    #    (i.e. walls of thickness >= k) exactly unchanged, including corners/T-junctions.
    if thickness > 1:
        k = _odd(thickness)
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

    # 2. Keep only long horizontal / vertical runs. Opening with a long 1px kernel removes
    #    everything that has no straight run of at least `line_len` in that direction.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
    walls = cv2.bitwise_or(
        cv2.morphologyEx(ink, cv2.MORPH_OPEN, h_kernel),
        cv2.morphologyEx(ink, cv2.MORPH_OPEN, v_kernel),
    )

    # 3. Close 1-2 px pinholes (e.g. where an H and a V run meet). 3x3 is far smaller than
    #    any door, so door openings are preserved.
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # 4. Connected-component filter: structural walls form long connected pieces; short
    #    isolated leftovers are clutter.
    return _remove_short_components(walls, min_extent)


def _remove_short_components(mask: np.ndarray, min_extent: int) -> np.ndarray:
    """Drop connected components whose bounding box's longest side is < min_extent."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    longest = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = longest >= min_extent
    keep[0] = False  # label 0 is the background
    return np.where(keep[labels], 255, 0).astype(np.uint8)
