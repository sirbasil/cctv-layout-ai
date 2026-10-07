"""Tests for the CCTV Layout AI FastAPI API.

Usage:
    python backend/test_api.py
    or
    pytest backend/test_api.py
"""
import sys
import time
import traceback
from pathlib import Path

from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND.parent))

from backend.api.app import app  # noqa: E402

client = TestClient(app)

SAMPLE_PNG = BACKEND / "tests" / "sample_floorplan.png"
SAMPLE_JPG = BACKEND / "tests" / "sample_floorplan.jpg"
SAMPLE_PDF = BACKEND / "tests" / "sample_floorplan.pdf"


def test_root_and_health():
    """Verify service root and health check endpoints."""
    r_root = client.get("/")
    assert r_root.status_code == 200, f"Root returned {r_root.status_code}"
    data_root = r_root.json()
    assert data_root["service"] == "cctv-layout-ai"
    assert data_root["status"] == "online"

    r_health = client.get("/health")
    assert r_health.status_code == 200, f"Health returned {r_health.status_code}"
    data_health = r_health.json()
    assert data_health["status"] == "healthy"


def test_analyze_sample_floorplan_png():
    """Verify POST /analyze on sample PNG produces baseline 6 cameras, >= 96% coverage, and complete assets."""
    with open(SAMPLE_PNG, "rb") as f:
        response = client.post(
            "/analyze",
            files={"file": ("sample_floorplan.png", f, "image/png")},
            data={
                "building_width_meters": 30.0,
                "target_coverage": 95.0,
                "max_cameras": 12,
                "fov_deg": 90.0,
                "max_range_m": 15.0,
                "wall_offset_m": 0.20,
                "include_rendered_images": "true",
            },
        )

    assert response.status_code == 200, f"Failed with {response.status_code}: {response.text}"
    data = response.json()

    # Top-level required fields
    assert data["success"] is True
    assert "coverage_percent" in data
    assert "camera_count" in data
    assert "uncovered_area_m2" in data
    assert "covered_area_m2" in data
    assert "total_interior_area_m2" in data
    assert "cameras" in data
    assert "blind_spots" in data
    assert "dimensions" in data
    assert "stop_reason" in data

    # Baseline regression checks
    assert data["camera_count"] <= 6, f"Too many cameras placed: {data['camera_count']}"
    assert data["coverage_percent"] >= 96.0, f"Coverage lower than baseline: {data['coverage_percent']}%"
    assert data["uncovered_area_m2"] <= 17.5, f"Uncovered area exceeds baseline: {data['uncovered_area_m2']}"

    # Verify camera structure
    cameras = data["cameras"]
    assert len(cameras) == data["camera_count"]
    for i, cam in enumerate(cameras, start=1):
        assert cam["id"] == i
        assert cam["x_px"] > 0
        assert cam["y_px"] > 0
        assert cam["x_m"] > 0
        assert cam["y_m"] > 0
        assert 0.0 <= cam["orientation_deg"] < 360.0
        assert cam["kind"] in ("wall", "corner")
        assert cam["added_coverage_percent"] > 0.0
        assert len(cam["fov_polygon"]) >= 3, "FOV polygon must have at least 3 vertices"
        for pt in cam["fov_polygon"]:
            assert len(pt) == 2

    # Verify blind spots
    blind_spots = data["blind_spots"]
    assert len(blind_spots) > 0
    for b in blind_spots:
        assert b["area_m2"] > 0
        assert b["priority"] in ("high", "medium", "low")
        assert b["size_class"] in ("large", "medium", "small")
        assert len(b["centroid_px"]) == 2
        assert len(b["bbox_px"]) == 4

    # Verify dimensions
    dims = data["dimensions"]
    assert dims["width_px"] == 1200
    assert dims["height_px"] == 825
    assert abs(dims["meters_per_pixel"] - 0.025) < 1e-4

    # Verify rendered visual assets (base64 data URIs)
    assert data["rendered_image_base64"] is not None
    assert data["rendered_image_base64"].startswith("data:image/png;base64,")
    assert len(data["rendered_image_base64"]) > 5000

    assert data["coverage_mask_base64"] is not None
    assert data["coverage_mask_base64"].startswith("data:image/png;base64,")

    assert data["blind_spot_mask_base64"] is not None
    assert data["blind_spot_mask_base64"].startswith("data:image/png;base64,")


def test_analyze_sample_floorplan_pdf():
    """Verify POST /analyze also works seamlessly with PDF floor plans."""
    with open(SAMPLE_PDF, "rb") as f:
        response = client.post(
            "/analyze",
            files={"file": ("sample_floorplan.pdf", f, "application/pdf")},
            data={
                "building_width_meters": 30.0,
                "target_coverage": 95.0,
                "max_cameras": 10,
                "include_rendered_images": "false",
            },
        )

    assert response.status_code == 200, f"PDF failed: {response.text}"
    data = response.json()
    assert data["camera_count"] <= 6
    assert data["coverage_percent"] >= 95.0
    # Verified include_rendered_images is false
    assert data["rendered_image_base64"] is None


def test_analyze_invalid_parameters():
    """Verify 400 Bad Request on invalid parameter bounds."""
    # 1. Invalid building_width_meters (<= 0)
    with open(SAMPLE_PNG, "rb") as f:
        r = client.post(
            "/analyze",
            files={"file": ("test.png", f, "image/png")},
            data={"building_width_meters": 0.0},
        )
    assert r.status_code == 400
    assert "building_width_meters" in r.json()["detail"]

    # 2. Invalid target_coverage (> 100)
    with open(SAMPLE_PNG, "rb") as f:
        r = client.post(
            "/analyze",
            files={"file": ("test.png", f, "image/png")},
            data={"building_width_meters": 30.0, "target_coverage": 150.0},
        )
    assert r.status_code == 400
    assert "target_coverage" in r.json()["detail"]

    # 3. Invalid max_cameras (< 1)
    with open(SAMPLE_PNG, "rb") as f:
        r = client.post(
            "/analyze",
            files={"file": ("test.png", f, "image/png")},
            data={"building_width_meters": 30.0, "max_cameras": 0},
        )
    assert r.status_code == 400
    assert "max_cameras" in r.json()["detail"]


def test_analyze_unsupported_format():
    """Verify 400 on unsupported file format (e.g. .txt)."""
    r = client.post(
        "/analyze",
        files={"file": ("notes.txt", b"Hello world", "text/plain")},
        data={"building_width_meters": 30.0},
    )
    assert r.status_code == 400
    assert "Unsupported file format" in r.json()["detail"]


def test_analyze_empty_file():
    """Verify 400 on 0-byte file."""
    r = client.post(
        "/analyze",
        files={"file": ("empty.png", b"", "image/png")},
        data={"building_width_meters": 30.0},
    )
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_analyze_corrupted_file():
    """Verify 400 on non-image corrupted file."""
    r = client.post(
        "/analyze",
        files={"file": ("broken.png", b"NOT AN IMAGE", "image/png")},
        data={"building_width_meters": 30.0},
    )
    assert r.status_code == 400
    assert "Preprocessing failed" in r.json()["detail"]


ALL_TESTS = [
    test_root_and_health,
    test_analyze_sample_floorplan_png,
    test_analyze_sample_floorplan_pdf,
    test_analyze_invalid_parameters,
    test_analyze_unsupported_format,
    test_analyze_empty_file,
    test_analyze_corrupted_file,
]


def main():
    passed = 0
    failed = 0
    for t in ALL_TESTS:
        name = t.__name__
        t0 = time.perf_counter()
        try:
            t()
            dt = time.perf_counter() - t0
            print(f"PASS  {name:<36} ({dt:.2f}s)")
            passed += 1
        except Exception as e:
            dt = time.perf_counter() - t0
            print(f"FAIL  {name:<36} ({dt:.2f}s)")
            traceback.print_exc()
            failed += 1

    print("==============================================")
    print(f"API test status: {'PASS' if failed == 0 else 'FAIL'}  ({passed}/{len(ALL_TESTS)})")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
