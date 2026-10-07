"""Pydantic schemas for the CCTV Layout AI API."""
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class CameraModelSchema(BaseModel):
    fov_deg: float = Field(..., description="Camera horizontal field of view in degrees")
    max_range_m: float = Field(..., description="Maximum effective camera range in meters")


class CameraSchema(BaseModel):
    id: int = Field(..., description="1-indexed camera identifier")
    x_px: int = Field(..., description="Camera X position in image pixels")
    y_px: int = Field(..., description="Camera Y position in image pixels")
    x_m: float = Field(..., description="Camera X position in real-world meters")
    y_m: float = Field(..., description="Camera Y position in real-world meters")
    orientation_deg: float = Field(
        ...,
        description="Camera pointing angle in degrees (0 = right, 90 = down, 180 = left, 270 = up)",
    )
    kind: str = Field(..., description="Candidate placement type: 'corner' or 'wall'")
    added_coverage_percent: float = Field(
        ...,
        description="Marginal interior coverage added by this camera alone (%)",
    )
    fov_deg: float = Field(..., description="FOV angle in degrees")
    max_range_m: float = Field(..., description="Max range in meters")
    range_px: float = Field(..., description="Max range in pixels")
    fov_polygon: List[List[int]] = Field(
        default_factory=list,
        description="Wall-clipped visibility polygon [[x, y], ...] in image pixels for vector rendering",
    )


class BlindSpotSchema(BaseModel):
    id: int = Field(..., description="1-indexed blind spot identifier")
    area_m2: float = Field(..., description="Uncovered area in square meters")
    area_px: int = Field(..., description="Uncovered area in image pixels")
    centroid_px: List[float] = Field(..., description="Centroid [x, y] in image pixels")
    centroid_m: List[float] = Field(..., description="Centroid [x, y] in real-world meters")
    bbox_px: List[int] = Field(..., description="Bounding box [x, y, w, h] in image pixels")
    bbox_m: List[float] = Field(..., description="Bounding box [x, y, w, h] in meters")
    priority: Literal["high", "medium", "low"] = Field(
        ..., description="Priority classification based on area: high (>=4m²), med (>=1m²), low (<1m²)"
    )
    size_class: Literal["large", "medium", "small"] = Field(
        ..., description="Size classification: large (>=4m²), med (>=1m²), small (<1m²)"
    )


class DimensionsSchema(BaseModel):
    width_px: int = Field(..., description="Normalized floor plan width in pixels")
    height_px: int = Field(..., description="Normalized floor plan height in pixels")
    meters_per_pixel: float = Field(..., description="Scale factor: meters per pixel")
    building_width_meters: float = Field(..., description="Total building width in meters")
    building_height_meters: float = Field(..., description="Total building height in meters")


class AnalysisResponse(BaseModel):
    success: bool = Field(True, description="Whether the analysis succeeded")
    coverage_percent: float = Field(..., description="Total interior coverage percentage achieved (0-100)")
    camera_count: int = Field(..., description="Total number of cameras selected")
    uncovered_area_m2: float = Field(..., description="Total uncovered interior area in m²")
    covered_area_m2: float = Field(..., description="Total covered interior area in m²")
    total_interior_area_m2: float = Field(..., description="Total interior floor area in m²")
    target_coverage: float = Field(..., description="Requested target coverage percentage")
    stop_reason: str = Field(..., description="Optimizer stopping reason")
    runtime_s: float = Field(..., description="Optimization computation runtime in seconds")
    dimensions: DimensionsSchema = Field(..., description="Image and real-world scale dimensions")
    camera_model: CameraModelSchema = Field(..., description="Camera optical parameters")
    cameras: List[CameraSchema] = Field(..., description="List of placed cameras with locations and FOVs")
    blind_spots: List[BlindSpotSchema] = Field(..., description="List of unmonitored interior regions")
    blind_spot_counts_by_priority: Dict[str, int] = Field(
        ..., description="Counts of blind spots partitioned by priority (high, medium, low)"
    )
    coverage_history: List[float] = Field(
        ..., description="Cumulative coverage percentage after each placed camera"
    )
    rendered_image_base64: Optional[str] = Field(
        None,
        description="Data URI (data:image/png;base64,...) of the annotated layout overlay",
    )
    coverage_mask_base64: Optional[str] = Field(
        None,
        description="Data URI of the binary coverage mask",
    )
    blind_spot_mask_base64: Optional[str] = Field(
        None,
        description="Data URI of the binary blind spot mask",
    )
