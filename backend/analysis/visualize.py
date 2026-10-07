"""Render optimizer results: coverage_mask.png, blind_spot_mask.png, and optimized_layout.png."""
import math
from pathlib import Path

import cv2
import numpy as np

from ..models.layout import CanonicalLayout
from .coverage import blind_spot_mask_image, coverage_mask_image
from .optimizer import OptimizationResult

# BGR palette
_COVERED = (120, 200, 60)       # green
_BLIND_HIGH = (30, 20, 235)     # bright red for high-priority blind spots
_BLIND_MED = (0, 140, 245)      # amber/orange for medium-priority blind spots
_BLIND_LOW = (90, 80, 190)      # muted crimson for low-priority blind spots
_WALL = (45, 35, 30)            # near-black
_CAM = (0, 165, 255)            # orange
_FOV_EDGE = (210, 130, 30)      # cyan/blue for wall-blocked FOV
_FOV_TINT = (245, 220, 180)     # very faint cyan/blue tint for visible cone
_TEXT_BG = (30, 30, 30)


def _blend(img, mask, color, alpha):
    if mask.any():
        img[mask] = (img[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


def render_layout(layout: CanonicalLayout, result: OptimizationResult) -> np.ndarray:
    """Floor plan + covered (green) + blind spots by priority + walls + cameras with wall-blocked FOV."""
    base = layout.image if layout.image is not None else np.full((layout.height, layout.width, 3), 255, np.uint8)
    gray = cv2.cvtColor(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    img = cv2.addWeighted(gray, 0.55, np.full_like(gray, 255), 0.45, 0)

    rep = result.report
    _blend(img, rep.covered_mask, _COVERED, 0.38)

    # Tint blind spots by priority
    for b in rep.blind_spots:
        x, y, w, h = b.bbox
        sub_mask = rep.blind_mask[y:y + h, x:x + w]
        color = _BLIND_HIGH if b.priority == "high" else (_BLIND_MED if b.priority == "medium" else _BLIND_LOW)
        alpha = 0.55 if b.priority == "high" else (0.45 if b.priority == "medium" else 0.35)
        if sub_mask.any():
            sub_img = img[y:y + h, x:x + w]
            sub_img[sub_mask] = (sub_img[sub_mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)

    img[layout.wall_mask > 0] = _WALL

    scale = max(layout.width, layout.height) / 1200.0
    r_cam = max(4, int(5 * scale))  # compact radius so marker stays cleanly off the wall
    arrow_len = max(16, int(26 * scale))
    th_box = max(1, int(scale))

    # 1. FOV visualization: wall-clipped visibility polygons ONLY (no unclipped geometric cones)
    inside = layout.building_mask > 0
    overlay = img.copy()
    for poly in result.fov_polygons:
        if len(poly) >= 3:
            cv2.fillPoly(overlay, [poly], _FOV_TINT)
    # Blend faint FOV wash only inside building
    overlay = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
    img[inside] = overlay[inside]

    # Draw sharp wall-clipped FOV outlines and boundary rays that stop at walls
    for poly in result.fov_polygons:
        if len(poly) >= 3:
            cv2.polylines(img, [poly], True, _FOV_EDGE, max(1, int(1.2 * scale)), cv2.LINE_AA)

    # 2. Blind-spot bounding boxes & priority annotations
    fs_s = 0.42 * scale
    for b in rep.blind_spots:
        x, y, w, h = b.bbox
        color = _BLIND_HIGH if b.priority == "high" else (_BLIND_MED if b.priority == "medium" else _BLIND_LOW)
        th = max(2, int(2 * scale)) if b.priority == "high" else max(1, int(scale))
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), color, th, cv2.LINE_AA)
        if b.priority in ("high", "medium") and w >= 25 and h >= 20:
            badge = f"{b.area_m2:.1f}m2 [{b.priority[0].upper()}]"
            (tw, tht), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, fs_s, th_box)
            bx = int(np.clip(x + 2, 2, layout.width - tw - 4))
            by = int(np.clip(y + tht + 2, tht + 2, layout.height - 4))
            cv2.rectangle(img, (bx - 1, by - tht - 1), (bx + tw + 1, by + 2), (255, 255, 255), -1)
            cv2.putText(img, badge, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, fs_s, color, th_box, cv2.LINE_AA)

    # 3. Cameras: inset mount circle + orientation arrow + label behind camera
    for n, cam in enumerate(result.cameras, 1):
        a = math.radians(cam.orientation_deg)
        tip = (int(cam.x + arrow_len * math.cos(a)), int(cam.y + arrow_len * math.sin(a)))
        cv2.arrowedLine(img, (cam.x, cam.y), tip, (20, 20, 20), max(2, int(2.5 * scale)), cv2.LINE_AA, tipLength=0.35)
        cv2.circle(img, (cam.x, cam.y), r_cam + 1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (cam.x, cam.y), r_cam, _CAM, -1, cv2.LINE_AA)

        # Label sits BEHIND the camera (opposite the arrow), clamped inside the image
        fs_l, th_l = 0.50 * scale, max(1, int(1.8 * scale))
        (tw, tht), _ = cv2.getTextSize(str(n), cv2.FONT_HERSHEY_SIMPLEX, fs_l, th_l)
        off = r_cam + 5 + max(tw, tht) / 2
        cx, cy = cam.x - off * math.cos(a), cam.y - off * math.sin(a)
        org = (int(np.clip(cx - tw / 2, 2, layout.width - tw - 2)),
               int(np.clip(cy + tht / 2, tht + 2, layout.height - 2)))
        for color, extra in (((255, 255, 255), 3), ((20, 20, 20), 0)):  # white halo, dark text
            cv2.putText(img, str(n), org, cv2.FONT_HERSHEY_SIMPLEX, fs_l, color, th_l + extra, cv2.LINE_AA)

    # 4. Header: key metrics + stop reason
    hp = rep.blind_spot_counts_by_priority
    lines = [
        f"Coverage {result.coverage_percent:.1f}%   Cameras {result.camera_count}   "
        f"Blind spots {len(rep.blind_spots)} ({rep.blind_area_m2:.1f} m2 uncovered: "
        f"{hp['high']} high, {hp['medium']} med, {hp['low']} low)",
        f"Camera model: FOV {result.camera_model.fov_deg:g} deg, range {result.camera_model.max_range_m:g} m   "
        f"(wall-geometry blocked, hackathon prototype)",
    ]
    fs, th = 0.5 * scale, max(1, int(scale))
    line_h = int(22 * scale)
    box_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][0] for t in lines) + 16
    cv2.rectangle(img, (0, 0), (box_w, line_h * len(lines) + 8), _TEXT_BG, -1)
    for k, t in enumerate(lines):
        cv2.putText(img, t, (8, line_h * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)

    # 5. Legend
    legend = [
        ("covered", _COVERED),
        ("blind [high]", _BLIND_HIGH),
        ("blind [med]", _BLIND_MED),
        ("wall", _WALL),
        ("camera", _CAM),
        ("wall-blocked FOV", _FOV_EDGE),
    ]
    x0, y0 = 8, layout.height - 10
    for label, color in legend:
        cv2.rectangle(img, (x0, y0 - int(12 * scale)), (x0 + int(14 * scale), y0), color, -1)
        cv2.putText(img, label, (x0 + int(18 * scale), y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale,
                    (20, 20, 20), th, cv2.LINE_AA)
        x0 += int(18 * scale) + cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, th)[0][0] + 16
    return img


def save_outputs(layout: CanonicalLayout, result: OptimizationResult, out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "coverage_mask": out / "coverage_mask.png",
        "blind_spot_mask": out / "blind_spot_mask.png",
        "optimized_layout": out / "optimized_layout.png",
    }
    cv2.imwrite(str(paths["coverage_mask"]), coverage_mask_image(layout, result.report))
    cv2.imwrite(str(paths["blind_spot_mask"]), blind_spot_mask_image(layout, result.report))
    cv2.imwrite(str(paths["optimized_layout"]), render_layout(layout, result))
    return paths
