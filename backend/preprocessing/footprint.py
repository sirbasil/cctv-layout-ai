"""Building footprint and interior free space.

    wall_mask ──copy──► temporary closed mask ──► exterior (flood from image border)
                                                      │
                          building_mask  = NOT exterior            (walls included)
                          interior_free  = building_mask AND NOT wall_mask

WHY A TEMPORARY CLOSED MASK
    A door in the outer wall (e.g. the main entrance) is a gap, so flood-filling from the
    image border over the raw wall mask would pour through that gap and mark the whole
    building as "exterior". To find the footprint we therefore bridge small gaps in a
    COPY of the wall mask. That copy is used ONLY to locate the exterior. The real
    wall_mask is never modified: door openings stay open in it, because the CCTV
    visibility stage needs the true line-of-sight geometry.

LIMITATIONS
  * Gaps wider than `max_gap_ratio` of the image's longer side are not bridged. An outer
    opening that wide (open facade, big garage door) lets the exterior leak into the room
    behind it, and that room is then missing from the footprint. This is NOT detected:
    preprocess() only warns when the interior comes out completely empty, so a PARTIAL
    leak is silent. If your plans have very wide outer openings, raise
    `footprint_gap_ratio` (the trade-off: more over-filling of narrow exterior notches).
  * Gaps are bridged along horizontal and vertical directions only (matching the
    horizontal/vertical wall assumption). A gap in a diagonal wall is not bridged.
  * Exterior pockets narrower than the bridging length (a very narrow light-well or
    courtyard notch) are absorbed into the footprint.
  * A frame drawn around the whole plan (page border) encloses the exterior and makes it
    look like part of the building. Page frames should be thin and are normally removed
    by the thickness filter in walls.py; a thick frame would not be.
  * The door gap itself is part of the footprint and of interior_free_mask (it lies on the
    wall line, between two interior sides), while wall_mask keeps it open.
"""
import cv2
import numpy as np


def _check_mask(mask: np.ndarray, name: str) -> None:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError(f"{name} must be a 2D numpy array (H, W)")


def build_building_mask(
    wall_mask: np.ndarray,
    max_gap_ratio: float = 0.10,
    min_component_ratio: float = 0.05,
) -> np.ndarray:
    """Return the building footprint: 255 = inside the building (walls included), 0 = exterior.

    `wall_mask` is NOT modified.

    Args:
        wall_mask: uint8 (H, W), 255 = wall.
        max_gap_ratio: widest wall gap (door) that is bridged when locating the exterior,
            as a fraction of the image's longer side. 0.10 bridges a 75 px entrance
            in a 1200 px wide plan with room to spare.
        min_component_ratio: separate footprint pieces smaller than this fraction of the
            largest piece (stray lines, outside clutter) are discarded.
    """
    _check_mask(wall_mask, "wall_mask")
    h, w = wall_mask.shape

    if not wall_mask.any():
        return np.zeros((h, w), np.uint8)

    # 1. Work on a COPY; the real wall mask must stay untouched.
    temp = (wall_mask > 0).astype(np.uint8) * 255

    # 2. Zero-pad BEFORE closing. OpenCV's erosion treats pixels outside the image as
    #    foreground, so without padding a long closing kernel would "bridge" the margin
    #    between a wall and the image edge and wrongly turn exterior into building. The
    #    padding also gives the flood fill a guaranteed exterior seed at (0, 0), even when
    #    the plan touches the image edge.
    gap = max(3, int(round(max(h, w) * max_gap_ratio)))
    gap += 1 - gap % 2  # odd size => centred anchor
    pad = gap
    padded = cv2.copyMakeBorder(temp, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    # 3. Temporarily bridge door-sized gaps. Closing is applied along each axis separately
    #    (a long 1-px kernel), so a gap in a horizontal wall is bridged by the horizontal
    #    kernel and a gap in a vertical wall by the vertical one - without blobbing walls
    #    together the way one big square kernel would.
    closed = cv2.bitwise_or(
        cv2.morphologyEx(padded, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 1))),
        cv2.morphologyEx(padded, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap))),
    )

    # 4. Exterior = the non-wall region connected to the (padded) image border.
    #    4-connectivity: the exterior must not squeeze diagonally between two wall pixels.
    _, labels = cv2.connectedComponents((closed == 0).astype(np.uint8), connectivity=4)
    exterior = (labels == labels[0, 0])[pad:-pad, pad:-pad]

    # 5. Footprint = everything that is not exterior (walls and enclosed rooms).
    building = np.where(exterior, 0, 255).astype(np.uint8)

    return _keep_significant_components(building, min_component_ratio)


def get_interior_free_mask(building_mask: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    """Return usable interior space: 255 = inside the building AND not wall, 0 = walls + exterior."""
    _check_mask(building_mask, "building_mask")
    _check_mask(wall_mask, "wall_mask")
    if building_mask.shape != wall_mask.shape:
        raise ValueError(f"shape mismatch: building {building_mask.shape} vs wall {wall_mask.shape}")
    inside = (building_mask > 0) & ~(wall_mask > 0)
    return np.where(inside, 255, 0).astype(np.uint8)


def _keep_significant_components(mask: np.ndarray, min_ratio: float) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    areas[0] = 0  # background
    keep = areas >= min_ratio * areas.max()
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)
