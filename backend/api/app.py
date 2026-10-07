"""FastAPI web application for CCTV Layout AI optimization service."""
import base64
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..analysis import (
    CameraModel,
    blind_spot_mask_image,
    coverage_mask_image,
    optimize_layout,
    render_layout,
)
from ..preprocessing import InvalidFileError, LayoutLoadError, UnsupportedFormatError, preprocess
from .schemas import (
    AnalysisResponse,
    BlindSpotSchema,
    CameraModelSchema,
    CameraSchema,
    DimensionsSchema,
)

app = FastAPI(
    title="CCTV Layout AI API",
    version="1.0.0",
    description="Automated CCTV camera placement optimization engine for architectural floor plans.",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_base64_png(bgr_or_gray: np.ndarray) -> str:
    """Encode OpenCV image array to base64 PNG data URI."""
    success, buffer = cv2.imencode(".png", bgr_or_gray)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64_str}"


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "cctv-layout-ai",
        "status": "online",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze": "POST /analyze",
        },
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "service": "cctv-layout-ai"}


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    tags=["Analysis"],
    summary="Analyze floor plan and optimize CCTV camera layout",
    description=(
        "Accepts a floor-plan image or PDF, building width in meters, target coverage, "
        "and camera constraints. Returns optimized camera placements, coverage metrics, "
        "blind spots, and rendering assets."
    ),
)
async def analyze_layout(
    file: UploadFile = File(..., description="Floor plan file: PNG, JPG, JPEG, or PDF"),
    building_width_meters: float = Form(..., description="Real-world building width in meters (must be > 0)"),
    target_coverage: float = Form(95.0, description="Desired interior coverage percentage (1.0 to 100.0)"),
    max_cameras: int = Form(12, description="Upper bound on number of cameras to place"),
    fov_deg: float = Form(90.0, description="Camera horizontal field of view in degrees"),
    max_range_m: float = Form(15.0, description="Camera maximum effective range in meters"),
    wall_offset_m: float = Form(0.20, description="Camera inset distance from walls in meters"),
    include_rendered_images: bool = Form(True, description="Whether to include base64-encoded visual renders"),
):
    # 1. Parameter Validation
    if building_width_meters is None or building_width_meters <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="building_width_meters must be a positive number greater than 0.",
        )
    if target_coverage <= 0.0 or target_coverage > 100.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="target_coverage must be between 1.0 and 100.0 percent.",
        )
    if max_cameras < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="max_cameras must be at least 1.",
        )
    if fov_deg <= 0.0 or fov_deg > 360.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="fov_deg must be between 1.0 and 360.0 degrees.",
        )
    if max_range_m <= 0.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="max_range_m must be a positive number greater than 0.",
        )

    # 2. Save uploaded file to temporary disk storage preserving extension
    original_filename = file.filename or "floorplan.png"
    suffix = Path(original_filename).suffix.lower()
    if not suffix:
        suffix = ".png"

    tmp_dir = tempfile.mkdtemp(prefix="cctv_upload_")
    tmp_path = Path(tmp_dir) / f"upload{suffix}"

    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )
        tmp_path.write_bytes(content)

        # 3. Preprocessing (frozen pipeline)
        try:
            layout = preprocess(tmp_path)
        except (UnsupportedFormatError, InvalidFileError, LayoutLoadError) as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Preprocessing failed: {str(err)}",
            )
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process floor plan: {str(err)}",
            )

        if not layout.interior_free_mask.any():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No enclosed interior space detected in the provided floor plan.",
            )

        # 4. Optimization (frozen optimizer)
        cam_model = CameraModel(fov_deg=fov_deg, max_range_m=max_range_m)
        try:
            result = optimize_layout(
                layout=layout,
                building_width_meters=building_width_meters,
                camera=cam_model,
                target_coverage=target_coverage,
                max_cameras=max_cameras,
                wall_offset_m=wall_offset_m,
            )
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Optimization algorithm error: {str(err)}",
            )

        # 5. Build structured Camera outputs
        mpp = result.meters_per_pixel
        range_px = cam_model.range_px(mpp)
        cameras_list = []
        for idx, cam in enumerate(result.cameras, start=1):
            poly = result.fov_polygons[idx - 1] if idx - 1 < len(result.fov_polygons) else np.zeros((0, 2))
            poly_coords = [[int(pt[0]), int(pt[1])] for pt in poly] if len(poly) > 0 else []
            cameras_list.append(
                CameraSchema(
                    id=idx,
                    x_px=int(cam.x),
                    y_px=int(cam.y),
                    x_m=round(float(cam.x_m), 2),
                    y_m=round(float(cam.y_m), 2),
                    orientation_deg=round(float(cam.orientation_deg), 1),
                    kind=cam.kind,
                    added_coverage_percent=round(float(cam.added_coverage_percent), 2),
                    fov_deg=round(float(cam_model.fov_deg), 1),
                    max_range_m=round(float(cam_model.max_range_m), 2),
                    range_px=round(float(range_px), 1),
                    fov_polygon=poly_coords,
                )
            )

        # 6. Build structured Blind Spot outputs
        rep = result.report
        blind_spots_list = []
        for idx, b in enumerate(rep.blind_spots, start=1):
            blind_spots_list.append(
                BlindSpotSchema(
                    id=idx,
                    area_m2=round(float(b.area_m2), 3),
                    area_px=int(b.area_px),
                    centroid_px=[round(float(c), 1) for c in b.centroid],
                    centroid_m=[round(float(c), 2) for c in b.centroid_m],
                    bbox_px=[int(v) for v in b.bbox],
                    bbox_m=[round(float(v), 2) for v in b.bbox_m],
                    priority=b.priority,
                    size_class=b.size_class,
                )
            )

        # 7. Render visualization masks if requested
        rendered_b64 = None
        cov_mask_b64 = None
        blind_mask_b64 = None
        if include_rendered_images:
            rendered_img = render_layout(layout, result)
            rendered_b64 = _to_base64_png(rendered_img)
            cov_mask_b64 = _to_base64_png(coverage_mask_image(layout, rep))
            blind_mask_b64 = _to_base64_png(blind_spot_mask_image(layout, rep))

        # 8. Assemble Dimensions and Final Response
        building_height_meters = layout.height * mpp
        dimensions = DimensionsSchema(
            width_px=int(layout.width),
            height_px=int(layout.height),
            meters_per_pixel=round(float(mpp), 5),
            building_width_meters=round(float(building_width_meters), 2),
            building_height_meters=round(float(building_height_meters), 2),
        )

        return AnalysisResponse(
            success=True,
            coverage_percent=round(float(result.coverage_percent), 2),
            camera_count=int(result.camera_count),
            uncovered_area_m2=round(float(rep.blind_area_m2), 2),
            covered_area_m2=round(float(rep.covered_area_m2), 2),
            total_interior_area_m2=round(float(rep.total_area_m2), 2),
            target_coverage=round(float(target_coverage), 1),
            stop_reason=result.stop_reason,
            runtime_s=round(float(result.runtime_s), 2),
            dimensions=dimensions,
            camera_model=CameraModelSchema(
                fov_deg=round(float(cam_model.fov_deg), 1),
                max_range_m=round(float(cam_model.max_range_m), 2),
            ),
            cameras=cameras_list,
            blind_spots=blind_spots_list,
            blind_spot_counts_by_priority=rep.blind_spot_counts_by_priority,
            coverage_history=[round(float(c), 2) for c in result.coverage_history],
            rendered_image_base64=rendered_b64,
            coverage_mask_base64=cov_mask_b64,
            blind_spot_mask_base64=blind_mask_b64,
        )

    finally:
        # Clean up temporary upload files
        shutil.rmtree(tmp_dir, ignore_errors=True)
