"""Test + visual-debug script for the preprocessing pipeline.

Usage (from the backend/ folder, or anywhere):
    python test_preprocessing.py                  # uses tests/sample_floorplan.png/.jpg/.pdf
    python test_preprocessing.py path/to/plan.pdf # your own PNG / JPG / PDF (generic checks only)

Saves (to backend/output/):
    normalized.png  wall_mask.png  free_mask.png  building_mask.png  interior_mask.png
    debug_layout.png   (exterior = black, interior = grey, walls = white)

Then LOOK at the images - that is the real test of whether the masks are usable.

Production code uses normal package imports (`backend.preprocessing`); this script puts the
PARENT of backend/ on sys.path so it can import the package the same way any caller would.
Ground-truth checks only run on the generated sample plan (they need its known geometry);
a custom input only gets the generic structural checks.
"""
import json
import subprocess
import sys
import tempfile
import traceback
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

BACKEND = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND.parent
sys.path.insert(0, str(PROJECT_ROOT))  # so `import backend...` works from any cwd

from backend.models.layout import CanonicalLayout  # noqa: E402
from backend.preprocessing import (  # noqa: E402
    InvalidFileError,
    UnsupportedFormatError,
    build_building_mask,
    extract_wall_mask,
    load_layout,
    normalize_image,
    preprocess,
)
from backend.tests import make_samples  # noqa: E402

OUTPUT_DIR = BACKEND / "output"
TESTS_DIR = BACKEND / "tests"
DEFAULT_SAMPLE = TESTS_DIR / "sample_floorplan.png"
FORMATS = ("png", "jpg", "pdf")

# Tolerances: generous on purpose. PNG / JPG / PDF differ by anti-aliasing and compression,
# so we never demand pixel-identical masks.
MIN_WALL_RECALL = 0.97  # share of (slightly eroded) true wall pixels found
MAX_SPURIOUS_WALL = 0.02  # share of detected wall pixels that are not near a true wall
MAX_FURNITURE_LEFT = 0.10  # wall px inside furniture bands / furniture outline px
MIN_IOU = 0.97  # building / interior vs ground truth
MAX_FORMAT_RATIO_DIFF = 0.15  # relative difference of wall_ratio between formats


# =============================================================================== helpers
class Tally:
    """Collects PASS/FAIL per named section so one failure doesn't hide the rest."""

    def __init__(self):
        self.failures = defaultdict(list)
        self.seen = []

    @contextmanager
    def section(self, *keys):
        for k in keys:
            if k not in self.seen:
                self.seen.append(k)
        try:
            yield
        except Exception as exc:  # noqa: BLE001 - we want every kind of failure reported
            detail = f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, AssertionError):
                detail += "\n" + traceback.format_exc()
            for k in keys:
                self.failures[k].append(detail)
            print(f"  FAIL [{' / '.join(keys)}] {detail}")

    def status(self, key):
        return "FAIL" if self.failures.get(key) else "PASS"


def count(mask):
    return int(np.count_nonzero(mask))


def iou(a, b):
    union = count((a > 0) | (b > 0))
    return count((a > 0) & (b > 0)) / union if union else 1.0


def _is_binary(mask):
    return bool(np.all((mask == 0) | (mask == 255)))


def debug_image(layout):
    """exterior = black, interior = grey, walls = white."""
    dbg = np.zeros((layout.height, layout.width, 3), np.uint8)
    dbg[layout.interior_free_mask > 0] = (150, 150, 150)
    dbg[layout.wall_mask > 0] = (255, 255, 255)
    return dbg


def save_outputs(layout):
    OUTPUT_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / "normalized.png"), layout.image)
    cv2.imwrite(str(OUTPUT_DIR / "wall_mask.png"), layout.wall_mask)
    cv2.imwrite(str(OUTPUT_DIR / "free_mask.png"), layout.free_space_mask)
    cv2.imwrite(str(OUTPUT_DIR / "building_mask.png"), layout.building_mask)
    cv2.imwrite(str(OUTPUT_DIR / "interior_mask.png"), layout.interior_free_mask)
    cv2.imwrite(str(OUTPUT_DIR / "debug_layout.png"), debug_image(layout))


def print_metrics(layout, path):
    print(f"Input:                      {path}")
    print(f"Image width:                {layout.width}")
    print(f"Image height:               {layout.height}")
    print(f"Source format:              {layout.source_format}")
    print(f"Wall pixel count:           {layout.wall_pixel_count}")
    print(f"Raw free-space pixel count: {layout.free_pixel_count}  (includes exterior)")
    print(f"Building pixel count:       {layout.building_pixel_count}")
    print(f"Interior free-space pixels: {layout.interior_free_pixel_count}")
    print(f"Wall ratio:                 {layout.wall_ratio:.3%}")
    print(f"Interior free-space ratio:  {layout.interior_free_ratio:.3%}")
    print(f"Saved to:                   {OUTPUT_DIR}/")


# =============================================================================== checks
def check_layout_contract(layout):
    """Structural contract every layout must satisfy (works for any input)."""
    shape = (layout.height, layout.width)
    for name in ("wall_mask", "free_space_mask", "building_mask", "interior_free_mask"):
        m = getattr(layout, name)
        assert isinstance(m, np.ndarray), f"{name} is not a numpy array"
        assert m.dtype == np.uint8, f"{name} dtype {m.dtype} != uint8"
        assert m.shape == shape, f"{name} shape {m.shape} != {shape}"
        assert _is_binary(m), f"{name} is not binary 0/255"
    assert layout.image is not None and layout.image.shape == (*shape, 3), "image missing / wrong shape"

    total = layout.width * layout.height
    assert layout.wall_pixel_count > 0, "no walls detected - mask is empty"
    assert layout.building_pixel_count > 0, "building_mask is empty"
    assert layout.interior_free_pixel_count > 0, "interior_free_mask is empty"
    assert layout.wall_pixel_count + layout.free_pixel_count == total, "wall/free masks don't partition image"
    assert layout.wall_ratio < 0.5, "over half the image is 'wall' - threshold probably failed"
    # raw free space still IS the plain inverse (the old behaviour is kept)
    assert not np.any((layout.free_space_mask > 0) & (layout.wall_mask > 0)), "free space overlaps walls"
    # interior invariants
    assert not np.any((layout.interior_free_mask > 0) & (layout.wall_mask > 0)), "interior overlaps walls"
    assert not np.any((layout.interior_free_mask > 0) & (layout.building_mask == 0)), "interior outside building"
    expected = layout.building_pixel_count - count((layout.building_mask > 0) & (layout.wall_mask > 0))
    assert layout.interior_free_pixel_count == expected, "interior != building AND NOT wall"
    # the interior must be strictly smaller than the raw free space (exterior excluded)
    assert layout.interior_free_pixel_count < layout.free_pixel_count, "interior includes the exterior"


def check_wall_quality(layout, gt, min_recall=MIN_WALL_RECALL):
    wall = layout.wall_mask
    # 5. major structural walls present
    core = cv2.erode(gt["wall"], np.ones((3, 3), np.uint8))
    recall = count((wall > 0) & (core > 0)) / count(core)
    assert recall >= min_recall, f"structural wall recall {recall:.3f} < {min_recall}"
    # nothing far from a true wall (furniture, dimension lines, text, circle ...)
    near_wall = cv2.dilate(gt["wall"], np.ones((5, 5), np.uint8))
    spurious = count((wall > 0) & (near_wall == 0)) / count(wall)
    assert spurious <= MAX_SPURIOUS_WALL, f"{spurious:.3%} of wall pixels are not near a real wall"
    # 6. furniture outlines substantially reduced
    if "furniture" in gt:
        left = count((wall > 0) & (gt["furniture"] > 0)) / count(gt["furniture_outline"])
        assert left <= MAX_FURNITURE_LEFT, f"furniture outlines survive in wall mask (ratio {left:.2f})"
    # 7. dimension lines substantially reduced
    if "dimension" in gt:
        dim_px = count((wall > 0) & (gt["dimension"] > 0))
        assert dim_px <= 20, f"{dim_px} wall px found on dimension line(s)"
    if "text" in gt:
        txt_px = count((wall > 0) & (gt["text"] > 0))
        assert txt_px <= 20, f"{txt_px} wall px found on text labels"


def check_interior_quality(layout, gt):
    b, i = layout.building_mask, layout.interior_free_mask
    # footprint / interior agree with the ideal ones
    assert iou(b, gt["footprint"]) >= MIN_IOU, f"building IoU {iou(b, gt['footprint']):.3f} < {MIN_IOU}"
    assert iou(i, gt["interior"]) >= MIN_IOU, f"interior IoU {iou(i, gt['interior']):.3f} < {MIN_IOU}"
    # 4. interior must not include the image exterior
    outside_gt = gt["footprint"] == 0
    leaked = count((i > 0) & outside_gt)
    assert leaked <= 0.002 * count(gt["interior"]), f"{leaked} interior px lie outside the building"
    for name, edge in (("top", i[0, :]), ("bottom", i[-1, :]), ("left", i[:, 0]), ("right", i[:, -1])):
        assert not edge.any(), f"interior touches the {name} image border (exterior leaked in)"


def check_doors_and_entrance(layout):
    """Door openings must stay OPEN in wall_mask and the entrance must not flood the interior."""
    s = layout.width / make_samples.W
    walls, *_ = make_samples.build_plan()
    wall, building, interior = layout.wall_mask, layout.building_mask, layout.interior_free_mask

    # interior door gaps: centre of the gap in the vertical wall x=600..616, gap y 300..380
    door_px = (round(608 * s), round(340 * s))  # (x, y)
    assert wall[door_px[1], door_px[0]] == 0, "interior door was closed in wall_mask"
    # entrance in the bottom wall (x 700..800, y 1030..1046)
    ex, ey = round(750 * s), round(1038 * s)
    assert wall[ey, ex] == 0, "entrance opening was closed in the final wall_mask"
    assert building[ey, ex] == 255, "entrance opening is not part of the footprint"
    # a point just OUTSIDE the entrance must be exterior, just INSIDE must be interior
    out_y, in_y = round(1070 * s), round(1000 * s)
    assert building[out_y, ex] == 0 and interior[out_y, ex] == 0, "area outside the entrance became building"
    assert interior[in_y, ex] == 255, "area just inside the entrance is not interior"
    # the outside strip below the building must be completely non-interior
    below = interior[round(1050 * s):, :]
    assert not below.any(), "exterior below the building is marked interior"


def check_footprint_does_not_modify_walls_and_needs_closing(layout):
    """Footprint detection must not touch wall_mask, and the temporary closing must matter."""
    wall = layout.wall_mask
    before = wall.copy()
    good = build_building_mask(wall)
    assert np.array_equal(wall, before), "build_building_mask modified the wall mask"

    # negative control: with (almost) no gap bridging the exterior leaks through the entrance
    # and the 'building' collapses to little more than the walls.
    leaky = build_building_mask(wall, max_gap_ratio=0.0)
    assert count(good) > 5 * count(leaky), (
        "disabling door bridging should make the exterior leak in - the test would "
        f"otherwise prove nothing (good={count(good)}, leaky={count(leaky)})"
    )


def check_stress_image(tally):
    """Crisp 1 px furniture / dimension lines + a thin legit wall (old code failed this)."""
    img, truth = make_samples.render_stress_image()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "stress.png"
        cv2.imwrite(str(p), img)
        layout = preprocess(p)

    gt = dict(truth)
    gt["interior"] = cv2.bitwise_and(truth["footprint"], cv2.bitwise_not(truth["wall"]))

    with tally.section("wall mask"):
        check_layout_contract(layout)
        check_wall_quality(layout, gt)
        kept = count((layout.wall_mask > 0) & (truth["partition"] > 0)) / count(truth["partition"])
        assert kept >= 0.9, f"legitimate thin partition wall was removed (kept {kept:.2f})"
        print(f"  stress image: walls OK, thin partition kept {kept:.0%}")
    with tally.section("interior mask"):
        check_interior_quality(layout, gt)
        print("  stress image: interior OK")


def check_serialization(layout):
    dumped = layout.model_dump()
    for name in ("wall_mask", "free_space_mask", "building_mask", "interior_free_mask", "image"):
        assert name not in dumped, f"{name} leaked into model_dump()"
    assert not any(isinstance(v, np.ndarray) for v in dumped.values()), "ndarray in model_dump()"
    text = layout.model_dump_json()
    assert len(text) < 2000, f"JSON is {len(text)} bytes - arrays probably serialized"
    data = json.loads(text)
    for key in ("wall_pixel_count", "free_pixel_count", "building_pixel_count",
                "interior_free_pixel_count", "wall_ratio", "interior_free_ratio"):
        assert key in data, f"metric '{key}' missing from serialization"
    assert data["interior_area_m2"] is None, "area should be None while scale is unknown"

    scaled = CanonicalLayout(**{**{k: getattr(layout, k) for k in (
        "source_format", "width", "height", "wall_mask", "free_space_mask",
        "building_mask", "interior_free_mask", "image")}, "meters_per_pixel": 0.02})
    assert abs(scaled.interior_area_m2 - scaled.interior_free_pixel_count * 0.02 ** 2) < 1e-9
    assert scaled.free_area_m2 > scaled.interior_area_m2, "raw free area should exceed interior area"


def check_model_validation(layout):
    """CanonicalLayout must reject masks that break the contract."""
    base = {k: getattr(layout, k) for k in (
        "source_format", "width", "height", "wall_mask", "free_space_mask",
        "building_mask", "interior_free_mask", "image")}

    def must_fail(label, **changes):
        try:
            CanonicalLayout(**{**base, **changes})
        except ValueError:  # pydantic ValidationError subclasses ValueError
            return
        raise AssertionError(f"CanonicalLayout accepted invalid input: {label}")

    must_fail("non-binary mask", wall_mask=np.full_like(layout.wall_mask, 128))
    must_fail("float mask", building_mask=layout.building_mask.astype(np.float32))
    must_fail("wrong shape", interior_free_mask=layout.interior_free_mask[:-1])
    must_fail("interior overlapping walls", interior_free_mask=np.full_like(layout.wall_mask, 255))
    must_fail("interior outside building", building_mask=np.zeros_like(layout.building_mask))


def check_imports_and_package():
    """Normal package imports must work from the project parent dir, in a fresh interpreter."""
    for stmt in ("import backend", "import backend.preprocessing", "import backend.models",
                 "from backend.preprocessing import preprocess"):
        r = subprocess.run([sys.executable, "-c", stmt], cwd=PROJECT_ROOT, capture_output=True, text=True)
        assert r.returncode == 0, f"`{stmt}` failed from {PROJECT_ROOT}:\n{r.stderr.strip()}"
    assert (BACKEND / "__init__.py").is_file(), "backend/__init__.py is missing"

    # production code must not use sys.path hacks or top-level `models` / `preprocessing` imports
    bad = []
    for py in list((BACKEND / "preprocessing").glob("*.py")) + list((BACKEND / "models").glob("*.py")):
        for n, line in enumerate(py.read_text().splitlines(), 1):
            code = line.split("#")[0].strip()
            if "sys.path" in code or code.startswith(("from models", "import models", "from preprocessing")):
                bad.append(f"{py.relative_to(BACKEND)}:{n}: {line.strip()}")
    assert not bad, "non-package imports / sys.path hacks in production code:\n  " + "\n  ".join(bad)


def check_loader_and_normalize():
    # every format -> ndarray, uint8, (H, W, 3), BGR
    for ext in FORMATS:
        p = TESTS_DIR / f"sample_floorplan.{ext}"
        img = load_layout(p)
        assert isinstance(img, np.ndarray) and img.dtype == np.uint8, f"{ext}: not a uint8 ndarray"
        assert img.ndim == 3 and img.shape[2] == 3, f"{ext}: shape {img.shape} is not (H, W, 3)"
    # BGR order: a pure-red PNG must come out as (0, 0, 255)
    with tempfile.TemporaryDirectory() as tmp:
        red = np.zeros((20, 30, 3), np.uint8)
        red[:, :] = (0, 0, 255)
        cv2.imwrite(str(Path(tmp) / "red.png"), red)
        px = load_layout(Path(tmp) / "red.png")[5, 5]
        assert tuple(int(v) for v in px) == (0, 0, 255), f"channel order is not BGR: {px}"

        # PDF: ONLY the first page is used
        import pymupdf
        doc = pymupdf.open()
        doc.new_page(width=400, height=200)  # page 1: 2:1 landscape
        doc.new_page(width=200, height=600)  # page 2: tall - must be ignored
        two = Path(tmp) / "two_pages.pdf"
        doc.save(str(two))
        doc.close()
        h, w = load_layout(two).shape[:2]
        assert abs(w / h - 2.0) < 0.05, f"PDF did not use the first page only ({w}x{h})"

    # normalize: aspect preserved, only shrinks, BGR uint8 out
    big = np.full((900, 2400, 3), 255, np.uint8)
    out = normalize_image(big, max_dimension=1200)
    assert out.shape == (450, 1200, 3) and out.dtype == np.uint8, f"unexpected normalized shape {out.shape}"
    small = np.full((300, 400, 3), 255, np.uint8)
    assert normalize_image(small, max_dimension=1200).shape == small.shape, "small image must not be resized"


def run_error_handling():
    """Bad inputs must raise clear errors, not crash with cryptic ones."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "notes.txt").write_text("hello")
        (tmp / "fake.png").write_bytes(b"this is not a png")
        (tmp / "fake.pdf").write_bytes(b"this is not a pdf")
        (tmp / "empty.jpg").write_bytes(b"")

        cases = [
            (tmp / "notes.txt", UnsupportedFormatError),
            (tmp / "fake.png", InvalidFileError),
            (tmp / "fake.pdf", InvalidFileError),
            (tmp / "empty.jpg", InvalidFileError),
            (tmp / "missing.png", InvalidFileError),
        ]
        for path, expected in cases:
            try:
                load_layout(path)
            except expected as exc:
                print(f"  {path.name:<12} -> {expected.__name__}: {exc}")
            else:
                raise AssertionError(f"{path.name} should have raised {expected.__name__}")
    print("Error handling:    PASS")


# =============================================================================== main
def main():
    custom = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if custom is None and not DEFAULT_SAMPLE.exists():
        make_samples.generate_samples()

    tally = Tally()
    # Any RuntimeWarning from the pipeline on the known-good sample is a regression.
    warnings.simplefilter("error", RuntimeWarning)

    print("=== package import / production-code hygiene ===")
    with tally.section("package import"):
        check_imports_and_package()
        print("  import backend.preprocessing from project parent: OK")

    print("\n=== preprocess() on main input ===")
    main_path = custom or DEFAULT_SAMPLE
    layout = None
    with tally.section("preprocessing status"):
        layout = preprocess(main_path)
        save_outputs(layout)
        print_metrics(layout, main_path)
        check_layout_contract(layout)
        print("Contract checks:            PASS")

    if layout is not None:
        with tally.section("preprocessing status"):
            check_serialization(layout)
            check_model_validation(layout)
            print("Serialization / validation: PASS")

    if custom is None:
        print("\n=== PNG / JPG / PDF: contract + ground truth ===")
        layouts = {}
        for ext in FORMATS:
            path = TESTS_DIR / f"sample_floorplan.{ext}"
            with tally.section(ext.upper()):
                lay = preprocess(path)
                layouts[ext] = lay
                check_layout_contract(lay)
                print(f"  {ext:<4} {lay.width}x{lay.height}  wall_ratio={lay.wall_ratio:.3%}  "
                      f"interior_ratio={lay.interior_free_ratio:.3%}")
            if ext in layouts:
                gt = make_samples.ground_truth(layouts[ext].width, layouts[ext].height)
                with tally.section(ext.upper(), "wall mask"):
                    check_wall_quality(layouts[ext], gt)
                with tally.section(ext.upper(), "interior mask"):
                    check_interior_quality(layouts[ext], gt)
                with tally.section(ext.upper(), "wall mask", "interior mask"):
                    check_doors_and_entrance(layouts[ext])

        print("\n=== format consistency (tolerant) ===")
        with tally.section("preprocessing status"):
            assert len(layouts) == 3, "not all formats preprocessed"
            base = layouts["png"]
            for ext, lay in layouts.items():
                assert (lay.width, lay.height) == (base.width, base.height), f"{ext}: size differs from png"
                diff = abs(lay.wall_ratio - base.wall_ratio) / base.wall_ratio
                assert diff < MAX_FORMAT_RATIO_DIFF, f"{ext} wall ratio differs {diff:.1%} from png"
                i = iou(lay.interior_free_mask, base.interior_free_mask)
                assert i >= MIN_IOU, f"{ext} interior mask IoU vs png is only {i:.3f}"
                print(f"  {ext:<4} wall-ratio diff {diff:.1%}, interior IoU vs png {i:.3f}")
            print("Format consistency: PASS")

        print("\n=== footprint behaviour ===")
        with tally.section("interior mask", "wall mask"):
            check_footprint_does_not_modify_walls_and_needs_closing(layouts["png"])
            print("  wall_mask untouched by footprint step; door bridging proven necessary: PASS")

        print("\n=== stress image (crisp thin furniture / dimension lines / thin wall) ===")
        check_stress_image(tally)

        print("\n=== loader + normalize ===")
        with tally.section("preprocessing status"):
            check_loader_and_normalize()
            print("  loader (PNG/JPG/PDF -> BGR uint8) and normalize: PASS")

    print("\n=== error handling ===")
    with tally.section("preprocessing status"):
        run_error_handling()

    # ---- summary ---------------------------------------------------------------------------
    keys = ["PNG", "JPG", "PDF", "wall mask", "interior mask", "package import"]
    overall_ok = all(tally.status(k) == "PASS" for k in keys) and not tally.failures.get("preprocessing status")
    print("\n" + "=" * 46)
    print(f"preprocessing status: {'PASS' if overall_ok else 'FAIL'}")
    for k in keys:
        print(f"{k + ':':<22}{tally.status(k) if custom is None or k in tally.seen else 'SKIPPED'}")
    print("=" * 46)
    if overall_ok:
        print("Now open output/wall_mask.png, building_mask.png, interior_mask.png, debug_layout.png.")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
