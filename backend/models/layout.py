"""Canonical layout: the single object the CCTV-analysis stage will consume.

Whatever the input was (PNG / JPG / PDF), preprocessing produces one CanonicalLayout.
NumPy arrays live on the model for fast in-process use but are EXCLUDED from
serialization, so `model_dump()` / `model_dump_json()` stay small (metadata + metrics).

MASK CONTRACT (all uint8, shape (height, width), values only 0 / 255)
    wall_mask           255 = structural wall / visibility obstacle (door openings stay open)
    free_space_mask     255 = RAW inverse of wall_mask. Includes the building exterior.
    building_mask       255 = building footprint (walls included), 0 = exterior
    interior_free_mask  255 = usable interior = building AND NOT wall. Use THIS for camera
                        candidate positions.
"""
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

_MASK_NAMES = ("wall_mask", "free_space_mask", "building_mask", "interior_free_mask")


class CanonicalLayout(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_format: Literal["png", "jpg", "pdf"]
    source_path: Optional[str] = None

    # Size of the NORMALIZED image/masks (what all later stages work in).
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    # Real-world scale of the normalized image. Unknown by default; needed later to
    # convert camera range (meters) into pixels. Set it once a scale is known.
    meters_per_pixel: Optional[float] = Field(default=None, gt=0)

    # uint8 (H, W) masks with values {0, 255}. Excluded from JSON on purpose.
    wall_mask: np.ndarray = Field(exclude=True, repr=False)
    free_space_mask: np.ndarray = Field(exclude=True, repr=False)
    building_mask: np.ndarray = Field(exclude=True, repr=False)
    interior_free_mask: np.ndarray = Field(exclude=True, repr=False)
    # Normalized BGR image, kept so later stages can draw coverage overlays on it.
    image: Optional[np.ndarray] = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def _check_arrays(self) -> "CanonicalLayout":
        expected = (self.height, self.width)
        for name in _MASK_NAMES:
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray) or arr.dtype != np.uint8:
                raise ValueError(f"{name} must be a uint8 numpy array")
            if arr.shape != expected:
                raise ValueError(f"{name} shape {arr.shape} != (height, width) {expected}")
            if not np.all((arr == 0) | (arr == 255)):
                raise ValueError(f"{name} must contain only 0 and 255")

        # The invariants the CCTV stage relies on: interior space is inside the building
        # and never overlaps a wall.
        if np.any((self.interior_free_mask > 0) & (self.wall_mask > 0)):
            raise ValueError("interior_free_mask overlaps wall_mask")
        if np.any((self.interior_free_mask > 0) & (self.building_mask == 0)):
            raise ValueError("interior_free_mask extends outside building_mask")

        if self.image is not None:
            if not isinstance(self.image, np.ndarray) or self.image.dtype != np.uint8:
                raise ValueError("image must be a uint8 numpy array")
            if self.image.ndim != 3 or self.image.shape != (*expected, 3):
                raise ValueError(f"image shape {self.image.shape} != {(*expected, 3)} (BGR)")
        return self

    # ---- derived metrics (included in serialization; cheap to recompute) ----------
    @computed_field
    @property
    def wall_pixel_count(self) -> int:
        return int(np.count_nonzero(self.wall_mask))

    @computed_field
    @property
    def free_pixel_count(self) -> int:
        """Raw inverse-wall free pixels (includes the exterior)."""
        return int(np.count_nonzero(self.free_space_mask))

    @computed_field
    @property
    def building_pixel_count(self) -> int:
        return int(np.count_nonzero(self.building_mask))

    @computed_field
    @property
    def interior_free_pixel_count(self) -> int:
        return int(np.count_nonzero(self.interior_free_mask))

    @computed_field
    @property
    def wall_ratio(self) -> float:
        """Wall pixels / total image pixels."""
        return self.wall_pixel_count / (self.width * self.height)

    @computed_field
    @property
    def interior_free_ratio(self) -> float:
        """Interior free pixels / total image pixels (same denominator as wall_ratio)."""
        return self.interior_free_pixel_count / (self.width * self.height)

    @computed_field
    @property
    def interior_area_m2(self) -> Optional[float]:
        """Usable interior area in square meters, or None if the scale is unknown."""
        if self.meters_per_pixel is None:
            return None
        return self.interior_free_pixel_count * self.meters_per_pixel**2

    @computed_field
    @property
    def free_area_m2(self) -> Optional[float]:
        """Area of the RAW inverse-wall free space in m^2, or None if the scale is unknown.

        WARNING: this includes the building exterior, so it is NOT a building-area metric.
        Use `interior_area_m2` for the area of the usable interior.
        """
        if self.meters_per_pixel is None:
            return None
        return self.free_pixel_count * self.meters_per_pixel**2
