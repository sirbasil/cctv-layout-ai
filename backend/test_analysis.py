"""Tests for the CCTV analysis stage (candidates, visibility, coverage, optimizer).

Usage (from anywhere):
    python backend/test_analysis.py

Uses the existing sample floor plan (through the real preprocessing pipeline) plus small
synthetic layouts with known geometry. Functions are named test_* so pytest can also collect
them, but pytest is not required.
"""
import sys
import tempfile
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND.parent))  # so `import backend...` works from any cwd

from backend.analysis import (  # noqa: E402
    CameraModel,
    classify_blind_spot,
    compute_coverage,
    generate_candidates,
    meters_per_pixel_from_width,
    optimize_layout,
    save_outputs,
    visible_points,
    wall_distance_map,
)
from backend.analysis.visibility import camera_coverage_mask  # noqa: E402
from backend.models.layout import CanonicalLayout  # noqa: E402
from backend.preprocessing import preprocess  # noqa: E402
from backend.tests import make_samples  # noqa: E402

SAMPLE = BACKEND / "tests" / "sample_floorplan.png"
WIDTH_M = 30.0
_CACHE = {}


# =============================================================================== fixtures
def sample_layout() -> CanonicalLayout:
    if "sample" not in _CACHE:
        if not SAMPLE.exists():
            make_samples.generate_samples()
        _CACHE["sample"] = preprocess(SAMPLE)
    return _CACHE["sample"]


def sample_result():
    if "result" not in _CACHE:
        _CACHE["result"] = optimize_layout(sample_layout(), building_width_meters=WIDTH_M)
    return _CACHE["result"]


def synthetic_layout(walls, outer=(10, 10, 190, 110), size=(200, 120)) -> CanonicalLayout:
    """Layout from wall rectangles (x1, y1, x2, y2, exclusive). Building = outer rectangle."""
    w, h = size
    wall = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in walls:
        wall[y1:y2, x1:x2] = 255
    building = np.zeros((h, w), np.uint8)
    x1, y1, x2, y2 = outer
    building[y1:y2, x1:x2] = 255
    interior = np.where((building > 0) & (wall == 0), 255, 0).astype(np.uint8)
    return CanonicalLayout(source_format="png", width=w, height=h, wall_mask=wall,
                           free_space_mask=255 - wall, building_mask=building,
                           interior_free_mask=interior)


def box_walls(x1=10, y1=10, x2=190, y2=110, t=4):
    return [(x1, y1, x2, y1 + t), (x1, y2 - t, x2, y2), (x1, y1, x1 + t, y2), (x2 - t, y1, x2, y2)]


def two_room_layout():
    """Closed box split by a SOLID wall at x=98..102 (no door): left / right room."""
    return synthetic_layout(box_walls() + [(98, 10, 102, 110)])


# =============================================================================== tests
def test_scale():
    lay = sample_layout()
    assert abs(meters_per_pixel_from_width(WIDTH_M, lay) - WIDTH_M / lay.width) < 1e-12
    for bad in (0, -5, None):
        try:
            meters_per_pixel_from_width(bad, lay)
        except ValueError:
            continue
        raise AssertionError(f"building_width_meters={bad} should be rejected")


def test_candidates_inside_interior_and_not_in_walls():
    for lay in (sample_layout(), two_room_layout()):
        mpp = WIDTH_M / lay.width
        cands = generate_candidates(lay, mpp)
        assert len(cands) > 0, "no candidates generated"
        xs = np.array([c.x for c in cands])
        ys = np.array([c.y for c in cands])
        assert (xs >= 0).all() and (xs < lay.width).all() and (ys >= 0).all() and (ys < lay.height).all()
        assert (lay.interior_free_mask[ys, xs] == 255).all(), "candidate outside interior_free_mask"
        assert (lay.wall_mask[ys, xs] == 0).all(), "candidate inside a wall"
        assert (lay.building_mask[ys, xs] == 255).all(), "candidate outside the building"
        assert {c.kind for c in cands} >= {"corner", "wall"}, "expected corner AND wall candidates"
    # wall-adjacent: every sample-plan candidate is within reasonable distance from a wall / edge
    lay = sample_layout()
    mpp = WIDTH_M / lay.width
    d = wall_distance_map(255 - lay.interior_free_mask)  # distance to non-interior
    far = [c for c in generate_candidates(lay, mpp) if d[c.y, c.x] > 1.5 / mpp]
    assert not far, f"{len(far)} candidates are more than 1.5 m from any wall"


def test_candidates_inset_into_interior_space():
    """Candidates and placed cameras must be cleanly inset into valid interior space."""
    lay = sample_layout()
    mpp = WIDTH_M / lay.width
    offset_m = 0.20
    cands = generate_candidates(lay, mpp, wall_offset_m=offset_m)
    assert len(cands) > 0, "no candidates generated"

    # Distance to nearest non-interior (wall or exterior)
    dist_interior = cv2.distanceTransform(lay.interior_free_mask, cv2.DIST_L2, 5)
    dist_wall = cv2.distanceTransform((lay.wall_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)

    expected_min_px = round(offset_m / mpp) - 1.0  # slight discretization margin (>= 7 px)
    for c in cands:
        # 1. Strictly inside interior_free_mask
        assert lay.interior_free_mask[c.y, c.x] == 255, f"Candidate ({c.x}, {c.y}) not in interior_free_mask"
        assert lay.wall_mask[c.y, c.x] == 0, f"Candidate ({c.x}, {c.y}) is inside a wall"
        assert lay.building_mask[c.y, c.x] == 255, f"Candidate ({c.x}, {c.y}) is outside building"
        # 2. Inset clearance from walls
        assert dist_wall[c.y, c.x] >= expected_min_px, (
            f"Candidate ({c.x}, {c.y}) clearance {dist_wall[c.y, c.x]:.1f} px < inset {expected_min_px:.1f} px"
        )
        # 3. Inset clearance from exterior
        assert dist_interior[c.y, c.x] >= expected_min_px, (
            f"Candidate ({c.x}, {c.y}) clearance {dist_interior[c.y, c.x]:.1f} px < inset {expected_min_px:.1f} px"
        )

    # Also verify optimizer placed cameras obey the inset requirement
    res = sample_result()
    for cam in res.cameras:
        assert lay.interior_free_mask[cam.y, cam.x] == 255
        assert lay.wall_mask[cam.y, cam.x] == 0
        assert dist_wall[cam.y, cam.x] >= expected_min_px - 1.0


def test_visibility_walls_fov_range():
    lay = two_room_layout()
    mpp = 0.1  # 200 px wide -> 20 m
    interior, wall = lay.interior_free_mask, lay.wall_mask
    cam = (30, 60)
    pts = np.array([[60, 60],   # same room, straight ahead
                    [150, 60],  # other room, behind the solid wall
                    [30, 30],   # same room, 90 deg off-axis (outside a 90 deg FOV)
                    [99, 60]])  # inside the wall itself
    model = CameraModel(fov_deg=90, max_range_m=15)
    vis = visible_points(cam, 0.0, pts, model, mpp, wall, interior)
    assert vis.tolist() == [True, False, False, False], f"unexpected visibility {vis.tolist()}"
    # range: same point becomes invisible with a 2 m range (point is 3 m away)
    vis_short = visible_points(cam, 0.0, pts[:1], CameraModel(90, 2.0), mpp, wall, interior)
    assert not vis_short[0], "point beyond max range was reported visible"
    # with a door in the dividing wall the other room becomes visible through the gap
    door = synthetic_layout(box_walls() + [(98, 10, 102, 50), (98, 70, 102, 110)])
    vis_door = visible_points(cam, 0.0, np.array([[150, 60]]), model, mpp,
                              door.wall_mask, door.interior_free_mask)
    assert vis_door[0], "point visible through an open door was reported hidden"


def test_fov_blocked_by_wall_geometry_not_geometric_cone():
    """Verify FOV coverage is actually blocked by wall geometry, not merely a geometric cone."""
    w, h = 200, 200
    mpp = 0.1  # 20 m across

    # 1. Solid wall dividing room in half at x = 100
    walls = [(0, 0, w, 5), (0, h - 5, w, h), (0, 0, 5, h), (w - 5, 0, w, h), (98, 5, 102, h - 5)]
    lay_walled = synthetic_layout(walls, outer=(5, 5, w - 5, h - 5), size=(w, h))

    # Identical layout without the dividing wall
    walls_open = [(0, 0, w, 5), (0, h - 5, w, h), (0, 0, 5, h), (w - 5, 0, w, h)]
    lay_open = synthetic_layout(walls_open, outer=(5, 5, w - 5, h - 5), size=(w, h))

    cam_model = CameraModel(fov_deg=90, max_range_m=15)
    origin = (30, 100)  # Camera in left room facing right (towards dividing wall)

    # Compute coverage for both
    dist_walled = wall_distance_map(lay_walled.wall_mask)
    mask_walled, poly_walled = camera_coverage_mask(origin, 0.0, cam_model, mpp,
                                                    lay_walled.interior_free_mask, dist_walled)

    dist_open = wall_distance_map(lay_open.wall_mask)
    mask_open, poly_open = camera_coverage_mask(origin, 0.0, cam_model, mpp,
                                                lay_open.interior_free_mask, dist_open)

    # Point at (150, 100) is in right room: inside geometric cone (dist=120px=12m <= 15m, angle=0)
    # With dividing wall: MUST BE BLOCKED
    assert not mask_walled[100, 150], "Point behind solid dividing wall was incorrectly covered!"
    assert np.count_nonzero(mask_walled[:, 100:]) == 0, "Coverage bled into the room behind the wall!"

    # Without dividing wall: MUST BE COVERED (proves wall alone caused occlusion)
    assert mask_open[100, 150], "Point in open room should be covered by geometric cone!"

    # Actual coverage with walls must be strictly smaller than unblocked cone
    assert np.count_nonzero(mask_walled) < np.count_nonzero(mask_open), (
        "Wall-blocked coverage must be strictly smaller than open space coverage"
    )

    # No wall pixel should ever be covered
    assert not np.any(mask_walled & (lay_walled.wall_mask > 0))

    # 2. Pillar shadow test: free-standing pillar in middle of room
    wall_pillar = np.zeros((h, w), np.uint8)
    wall_pillar[:5, :] = 255
    wall_pillar[-5:, :] = 255
    wall_pillar[:, :5] = 255
    wall_pillar[:, -5:] = 255
    wall_pillar[80:120, 70:90] = 255  # Pillar at x=70..90, y=80..120
    bldg = np.zeros((h, w), np.uint8)
    bldg[5:-5, 5:-5] = 255
    int_pillar = np.where((bldg > 0) & (wall_pillar == 0), 255, 0).astype(np.uint8)
    lay_pillar = CanonicalLayout(source_format="png", width=w, height=h, wall_mask=wall_pillar,
                                 free_space_mask=255 - wall_pillar, building_mask=bldg,
                                 interior_free_mask=int_pillar)

    mask_pillar, _ = camera_coverage_mask((20, 100), 0.0, cam_model, mpp, int_pillar,
                                          wall_distance_map(wall_pillar))
    # In front of pillar: covered
    assert np.count_nonzero(mask_pillar[90:110, 30:60]) > 0
    # Shadow directly behind pillar (x=120..180, y=90..110): 100% blocked
    shadow_px = np.count_nonzero(mask_pillar[90:110, 120:180])
    assert shadow_px == 0, f"Pillar shadow must be completely uncovered, found {shadow_px} px"


def test_coverage_bounds_and_area_partition():
    lay = sample_layout()
    mpp = WIDTH_M / lay.width
    model = CameraModel()
    rng = np.random.default_rng(0)
    cands = generate_candidates(lay, mpp)
    for n in (0, 1, 3, 8):
        picks = rng.choice(len(cands), size=n, replace=False) if n else []
        cams = [(cands[i].x, cands[i].y, float(rng.uniform(0, 360))) for i in picks]
        rep = compute_coverage(lay, cams, model, mpp)
        assert 0.0 <= rep.coverage_percent <= 100.0, f"coverage {rep.coverage_percent} out of range"
        assert rep.total_area_px == lay.interior_free_pixel_count
        assert rep.covered_area_px + rep.blind_area_px == rep.total_area_px
        assert abs(rep.total_area_m2 - lay.interior_free_pixel_count * mpp ** 2) < 1e-6
        if n == 0:
            assert rep.coverage_percent == 0.0 and rep.blind_area_px == rep.total_area_px


def test_adding_cameras_never_reduces_coverage():
    lay = sample_layout()
    mpp = WIDTH_M / lay.width
    model = CameraModel()
    res = sample_result()
    cams = [(c.x, c.y, c.orientation_deg) for c in res.cameras]
    prev = -1.0
    for k in range(len(cams) + 1):
        cov = compute_coverage(lay, cams[:k], model, mpp).coverage_percent
        assert cov >= prev - 1e-9, f"coverage dropped from {prev:.3f} to {cov:.3f} at camera {k}"
        prev = cov
    assert all(b >= a for a, b in zip(res.coverage_history, res.coverage_history[1:]))
    # random (non-greedy) additions too
    rng = np.random.default_rng(1)
    cands = generate_candidates(lay, mpp)
    chosen, prev = [], 0.0
    for i in rng.choice(len(cands), size=6, replace=False):
        chosen.append((cands[i].x, cands[i].y, float(rng.uniform(0, 360))))
        cov = compute_coverage(lay, chosen, model, mpp).coverage_percent
        assert cov >= prev - 1e-9, "adding a random camera reduced coverage"
        prev = cov


def test_blind_spots_exclude_exterior_and_walls():
    lay = sample_layout()
    rep = sample_result().report
    for mask_name, mask in (("blind", rep.blind_mask), ("covered", rep.covered_mask)):
        assert not np.any(mask & (lay.building_mask == 0)), f"{mask_name} area includes the exterior"
        assert not np.any(mask & (lay.wall_mask > 0)), f"{mask_name} area includes walls"
    for b in rep.blind_spots:
        x, y, w, h = b.bbox
        assert lay.building_mask[y:y + h, x:x + w].any(), "blind spot outside the building"
    # zero cameras: blind area == entire interior and nothing outside it
    zero = compute_coverage(lay, [], CameraModel(), WIDTH_M / lay.width)
    assert np.array_equal(zero.blind_mask, lay.interior_free_mask > 0)


def test_blind_spots_regions_and_classification():
    """Blind spots must be returned as regions with area, centroid, bbox, and priority/size."""
    res = sample_result()
    rep = res.report

    # Classification function unit checks
    assert classify_blind_spot(5.0) == ("high", "large")
    assert classify_blind_spot(4.0) == ("high", "large")
    assert classify_blind_spot(2.5) == ("medium", "medium")
    assert classify_blind_spot(1.0) == ("medium", "medium")
    assert classify_blind_spot(0.5) == ("low", "small")

    assert len(rep.blind_spots) > 0, "expected at least one blind spot on sample layout"
    for b in rep.blind_spots:
        # Area checks
        assert b.area_px > 0, "blind spot area_px must be positive"
        assert b.area_m2 > 0, "blind spot area_m2 must be positive"
        assert abs(b.area_m2 - b.area_px * res.meters_per_pixel ** 2) < 1e-5

        # Centroid checks
        cx, cy = b.centroid
        assert 0 <= cx < sample_layout().width
        assert 0 <= cy < sample_layout().height
        assert len(b.centroid_m) == 2

        # Bounding box checks
        bx, by, bw, bh = b.bbox
        assert bw > 0 and bh > 0
        assert bx >= 0 and by >= 0
        assert bx + bw <= sample_layout().width
        assert by + bh <= sample_layout().height
        assert len(b.bbox_m) == 4

        # Classification checks
        assert b.priority in {"high", "medium", "low"}
        assert b.size_class in {"large", "medium", "small"}
        if b.area_m2 >= 4.0:
            assert b.priority == "high" and b.size_class == "large"
        elif b.area_m2 >= 1.0:
            assert b.priority == "medium" and b.size_class == "medium"
        else:
            assert b.priority == "low" and b.size_class == "small"

        # Serialization dictionary checks
        d = b.to_dict()
        for k in ("area_m2", "area_px", "centroid", "centroid_m", "bbox", "bbox_m", "priority", "size_class"):
            assert k in d, f"missing key '{k}' in blind spot dictionary"

    # Report aggregation checks
    hp = rep.blind_spot_counts_by_priority
    hs = rep.blind_spot_counts_by_size
    assert sum(hp.values()) == len(rep.blind_spots)
    assert sum(hs.values()) == len(rep.blind_spots)


def test_optimizer_on_sample_and_terminates():
    lay = sample_layout()
    wall_before = lay.wall_mask.copy()
    res = sample_result()
    assert np.array_equal(lay.wall_mask, wall_before), "optimizer modified wall_mask"
    assert 1 <= res.camera_count <= 20
    assert res.coverage_percent >= 95.0 or res.camera_count == 20 or "no candidate" in res.stop_reason
    assert res.coverage_percent >= 90.0, f"sample plan coverage unexpectedly low ({res.coverage_percent:.1f}%)"
    for c in res.cameras:
        assert lay.interior_free_mask[c.y, c.x] == 255 and lay.wall_mask[c.y, c.x] == 0
        assert 0.0 <= c.orientation_deg < 360.0
    # reported coverage == independent recomputation from the returned cameras
    again = compute_coverage(lay, [(c.x, c.y, c.orientation_deg) for c in res.cameras],
                             res.camera_model, res.meters_per_pixel)
    assert abs(again.coverage_percent - res.coverage_percent) < 1e-6, "coverage not reproducible"
    d = res.to_dict()
    assert d["camera_count"] == res.camera_count and len(d["cameras"]) == res.camera_count
    assert "blind_spot_counts_by_priority" in d["coverage"]
    assert "blind_spot_counts_by_size" in d["coverage"]

    # termination: camera cap, unreachable target, zero target, empty interior
    t0 = time.perf_counter()
    capped = optimize_layout(lay, WIDTH_M, target_coverage=100.0, max_cameras=3)
    assert capped.camera_count <= 3 and "limit" in capped.stop_reason
    unreachable = optimize_layout(lay, WIDTH_M, camera=CameraModel(30, 3), target_coverage=100.0, max_cameras=20)
    assert unreachable.camera_count <= 20
    none = optimize_layout(lay, WIDTH_M, target_coverage=0.0)
    assert none.camera_count == 0
    empty = synthetic_layout([(0, 0, 200, 120)], outer=(0, 0, 200, 120))  # all wall, no interior
    e = optimize_layout(empty, 20.0)
    assert e.camera_count == 0 and e.coverage_percent == 0.0
    assert time.perf_counter() - t0 < 120, "optimizer too slow"


def test_sample_floorplan_baseline_regression():
    """Regression test: sample floor plan must achieve >= 96.0% coverage with <= 6 cameras."""
    lay = sample_layout()
    res = optimize_layout(lay, building_width_meters=30.0, target_coverage=95.0)
    assert res.camera_count <= 6, f"Regression: expected <= 6 cameras, got {res.camera_count}"
    assert res.coverage_percent >= 96.0, f"Regression: expected >= 96.0% coverage, got {res.coverage_percent:.2f}%"
    assert res.report.blind_area_m2 <= 17.5, (
        f"Regression: expected <= 17.5 m2 uncovered, got {res.report.blind_area_m2:.2f} m2"
    )


def test_two_rooms_need_two_cameras():
    """A solid wall blocks sight, so each closed room needs its own camera."""
    lay = two_room_layout()
    res = optimize_layout(lay, 20.0, camera=CameraModel(90, 15), target_coverage=90.0)
    xs = [c.x for c in res.cameras]
    assert any(x < 98 for x in xs) and any(x > 102 for x in xs), f"cameras {xs} not in both rooms"
    assert res.coverage_percent >= 90.0


def test_outputs_saved():
    with tempfile.TemporaryDirectory() as tmp:
        paths = save_outputs(sample_layout(), sample_result(), tmp)
        for name in ("coverage_mask", "blind_spot_mask", "optimized_layout"):
            assert name in paths, f"Missing output path for {name}"
            p = paths[name]
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            assert img is not None and img.shape[:2] == (sample_layout().height, sample_layout().width), name


# =============================================================================== runner
def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        t0 = time.perf_counter()
        try:
            t()
            print(f"PASS  {t.__name__}  ({time.perf_counter() - t0:.1f}s)")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}\n{traceback.format_exc()}")
    print("=" * 46)
    print(f"analysis status: {'PASS' if not failed else 'FAIL'}  ({len(tests) - failed}/{len(tests)})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
