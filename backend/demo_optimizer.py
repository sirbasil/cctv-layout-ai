"""Demo: sample floor plan -> CanonicalLayout -> optimized CCTV layout.

Usage (from anywhere):
    python backend/demo_optimizer.py                                  # sample plan, 30 m wide
    python backend/demo_optimizer.py path/to/plan.pdf --width 42.5    # your plan
    python backend/demo_optimizer.py --fov 110 --range 12 --target 90 --max-cameras 15

Saves backend/output/coverage_mask.png, blind_spot_mask.png, optimized_layout.png and optimization.json.
Prototype estimate only - not hardware-grade CCTV engineering.
"""
import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND.parent))  # so `import backend...` works from any cwd

from backend.analysis import CameraModel, optimize_layout, save_outputs  # noqa: E402
from backend.preprocessing import preprocess  # noqa: E402

DEFAULT_SAMPLE = BACKEND / "tests" / "sample_floorplan.png"
OUTPUT_DIR = BACKEND / "output"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", nargs="?", default=str(DEFAULT_SAMPLE), help="PNG / JPG / JPEG / PDF floor plan")
    ap.add_argument("--width", type=float, default=30.0,
                    help="building_width_meters: real width represented by the normalized image width")
    ap.add_argument("--fov", type=float, default=90.0, help="camera horizontal FOV, degrees")
    ap.add_argument("--range", type=float, default=15.0, help="camera max range, meters")
    ap.add_argument("--offset", type=float, default=0.20,
                    help="wall_offset_m: distance (meters) camera positions are inset into interior space")
    ap.add_argument("--target", type=float, default=95.0, help="target coverage, percent")
    ap.add_argument("--max-cameras", type=int, default=20)
    args = ap.parse_args(argv)

    layout = preprocess(args.plan)
    result = optimize_layout(layout, building_width_meters=args.width,
                             camera=CameraModel(fov_deg=args.fov, max_range_m=args.range),
                             wall_offset_m=args.offset,
                             target_coverage=args.target, max_cameras=args.max_cameras)
    paths = save_outputs(layout, result, OUTPUT_DIR)
    (OUTPUT_DIR / "optimization.json").write_text(json.dumps(result.to_dict(), indent=2))

    rep = result.report
    hp = rep.blind_spot_counts_by_priority
    print(f"Plan:        {args.plan}  ({layout.width}x{layout.height} px, "
          f"{result.meters_per_pixel * 100:.2f} cm/px)")
    print(f"Interior:    {rep.total_area_m2:.1f} m2   candidates: {result.candidate_count}   "
          f"samples: {result.sample_count}")
    print()
    print("Selected cameras:")
    for n, c in enumerate(result.cameras, 1):
        print(f"  Camera {n} → +{c.added_coverage_percent:4.1f}%  ({c.x_m:4.1f}m, {c.y_m:4.1f}m)  "
              f"facing {c.orientation_deg:5.1f}°  [{c.kind:<6}]")
    print()
    print(f"Coverage: {result.coverage_percent:.1f}%")
    print(f"Cameras: {result.camera_count}")
    print(f"Uncovered: {rep.blind_area_m2:.1f} m²")
    print(f"Blind spots: {len(rep.blind_spots)}  (High: {hp['high']}, Medium: {hp['medium']}, Low: {hp['low']})")
    print()
    print(f"Stop reason: {result.stop_reason}   ({result.runtime_s:.1f} s)")
    print(f"Saved: {paths['optimized_layout']}\n       {paths['coverage_mask']}\n       {paths['blind_spot_mask']}")


if __name__ == "__main__":
    main()
