"""Generate a synthetic floor plan as PNG, JPG and (vector) PDF for quick testing.

The plan deliberately includes things that make wall detection hard: door gaps,
thin furniture outlines, text labels, and a long thin dimension line.
Run directly:  python tests/make_samples.py
"""
from pathlib import Path

import cv2
import numpy as np
import pymupdf

HERE = Path(__file__).parent
W, H = 1600, 1100
T = 16  # wall thickness (px / pt)


def _wall(x1, y1, x2, y2, gaps=(), horizontal=True):
    """Return wall rects for a straight wall with door gaps [(start, end), ...]."""
    a, b = (x1, x2) if horizontal else (y1, y2)
    cuts, pos = [], a
    for g0, g1 in sorted(gaps):
        cuts.append((pos, g0))
        pos = g1
    cuts.append((pos, b))
    if horizontal:
        return [(s, y1, e, y1 + T) for s, e in cuts if e > s]
    return [(x1, s, x1 + T, e) for s, e in cuts if e > s]


def build_plan():
    walls = []
    # outer walls (entrance gap in the bottom wall)
    walls += _wall(60, 60, 1540, 60, horizontal=True)
    walls += _wall(60, 1030, 1540, 1030, gaps=[(700, 800)], horizontal=True)
    walls += _wall(60, 60, 60, 1030, horizontal=False)
    walls += _wall(1530, 60, 1530, 1030, horizontal=False)
    # interior walls with doors
    walls += _wall(600, 60, 600, 1030, gaps=[(300, 380), (760, 840)], horizontal=False)
    walls += _wall(1100, 60, 1100, 600, gaps=[(250, 330)], horizontal=False)
    walls += _wall(600, 600, 1530, 600, gaps=[(800, 880), (1250, 1330)], horizontal=True)
    walls += _wall(60, 500, 600, 500, gaps=[(250, 330)], horizontal=True)
    furniture = [(120, 120, 280, 200), (320, 130, 420, 190), (700, 700, 900, 780),
                 (1200, 120, 1380, 220), (700, 120, 760, 400), (150, 580, 330, 700)]
    circles = [(900, 300, 60)]
    labels = [("LIVING", 160, 420), ("OFFICE", 700, 520), ("BEDROOM", 1200, 540),
              ("KITCHEN", 160, 960)]
    dim_lines = [(60, 30, 1540, 30)]  # long thin dimension line above the plan
    return walls, furniture, circles, labels, dim_lines


def render_png_jpg():
    walls, furniture, circles, labels, dims = build_plan()
    img = np.full((H, W, 3), 255, np.uint8)
    for x1, y1, x2, y2 in walls:
        cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 0), -1)
    for x1, y1, x2, y2 in furniture:
        cv2.rectangle(img, (x1, y1), (x2, y2), (90, 90, 90), 2)
    for cx, cy, r in circles:
        cv2.circle(img, (cx, cy), r, (90, 90, 90), 2)
    for txt, x, y in labels:
        cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 60, 60), 2)
    for x1, y1, x2, y2 in dims:
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 1)
    return img


def render_pdf(path):
    walls, furniture, circles, labels, dims = build_plan()
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    for x1, y1, x2, y2 in walls:
        page.draw_rect(pymupdf.Rect(x1, y1, x2, y2), color=None, fill=(0, 0, 0))
    for x1, y1, x2, y2 in furniture:
        page.draw_rect(pymupdf.Rect(x1, y1, x2, y2), color=(0.35,) * 3, width=2)
    for cx, cy, r in circles:
        page.draw_circle(pymupdf.Point(cx, cy), r, color=(0.35,) * 3, width=2)
    for txt, x, y in labels:
        page.insert_text(pymupdf.Point(x, y), txt, fontsize=40, color=(0.25,) * 3)
    for x1, y1, x2, y2 in dims:
        page.draw_line(pymupdf.Point(x1, y1), pymupdf.Point(x2, y2), color=(0, 0, 0), width=1)
    doc.save(str(path))
    doc.close()


def generate_samples():
    img = render_png_jpg()
    cv2.imwrite(str(HERE / "sample_floorplan.png"), img)
    cv2.imwrite(str(HERE / "sample_floorplan.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    render_pdf(HERE / "sample_floorplan.pdf")
    return [HERE / f"sample_floorplan.{e}" for e in ("png", "jpg", "pdf")]


# --------------------------------------------------------------------------- ground truth
def _scaled_rect_mask(shape, rects, s, thickness=-1):
    m = np.zeros(shape, np.uint8)
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(m, (round(x1 * s), round(y1 * s)), (round(x2 * s) - 1, round(y2 * s) - 1), 255, thickness)
    return m


def ground_truth(width, height):
    """Ideal masks for the sample plan at the NORMALIZED size (width x height).

    Returns a dict of uint8 {0,255} masks:
      wall        exact wall rectangles
      footprint   outer rectangle of the building (walls included)
      interior    footprint minus walls
      furniture   thick band around furniture outlines (+ the circle)
      furniture_outline  the outlines themselves, 1 px (to normalise "how much survived")
      text        boxes around the room labels
      dimension   band around the long dimension line
    """
    s = width / W
    shape = (height, width)
    walls, furniture, circles, labels, dims = build_plan()

    wall = _scaled_rect_mask(shape, walls, s)
    xs = [x for x1, _, x2, _ in walls for x in (x1, x2)]
    ys = [y for _, y1, _, y2 in walls for y in (y1, y2)]
    footprint = _scaled_rect_mask(shape, [(min(xs), min(ys), max(xs), max(ys))], s)
    interior = cv2.bitwise_and(footprint, cv2.bitwise_not(wall))

    band = _scaled_rect_mask(shape, furniture, s, thickness=7)
    outline = _scaled_rect_mask(shape, furniture, s, thickness=1)
    for cx, cy, r in circles:
        c = (round(cx * s), round(cy * s))
        cv2.circle(band, c, round(r * s), 255, 7)
        cv2.circle(outline, c, round(r * s), 255, 1)

    text = np.zeros(shape, np.uint8)
    for txt, x, y in labels:
        (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        x0, y0, x1, y1 = x - 6, y - th - 6, x + tw + 6, y + base + 6
        cv2.rectangle(text, (round(x0 * s), round(y0 * s)), (round(x1 * s), round(y1 * s)), 255, -1)

    dimension = np.zeros(shape, np.uint8)
    for x1, y1, x2, y2 in dims:
        cv2.rectangle(dimension, (round(x1 * s), round(y1 * s) - 5), (round(x2 * s), round(y2 * s) + 5), 255, -1)

    return {"wall": wall, "footprint": footprint, "interior": interior,
            "furniture": band, "furniture_outline": outline, "text": text, "dimension": dimension}


# --------------------------------------------------------------------------- stress image
def render_stress_image(width=1200):
    """A HARDER, crisp version of the sample, rendered directly at the working size.

    Differences from the regular sample (which is rendered at 1600 px and then blurred by
    the 0.75 downscale, so its thin lines happen to fall below the Otsu threshold):
      * furniture, circle and text are PURE BLACK 1 px lines (no anti-aliasing luck)
      * two long 1 px black dimension lines (top and right) with end ticks
      * one legitimate THIN partition wall (5 px) inside the big room
    Returns (image, truth) where truth has the ideal wall mask and the line bands.
    """
    s = width / W
    height = round(H * s)
    walls, furniture, circles, labels, _ = build_plan()
    img = np.full((height, width, 3), 255, np.uint8)
    black = (0, 0, 0)

    for x1, y1, x2, y2 in walls:
        cv2.rectangle(img, (round(x1 * s), round(y1 * s)), (round(x2 * s) - 1, round(y2 * s) - 1), black, -1)
    for x1, y1, x2, y2 in furniture:
        cv2.rectangle(img, (round(x1 * s), round(y1 * s)), (round(x2 * s), round(y2 * s)), black, 1)
    for cx, cy, r in circles:
        cv2.circle(img, (round(cx * s), round(cy * s)), round(r * s), black, 1)
    for txt, x, y in labels:
        cv2.putText(img, txt, (round(x * s), round(y * s)), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * s, black, 1)

    # dimension lines OUTSIDE the building (building spans x 45..1160, y 45..785 at 1200 px)
    top_y, right_x = 20, 1182
    cv2.line(img, (45, top_y), (1160, top_y), black, 1)
    cv2.line(img, (45, top_y - 5), (45, top_y + 5), black, 1)
    cv2.line(img, (1160, top_y - 5), (1160, top_y + 5), black, 1)
    cv2.line(img, (right_x, 45), (right_x, 785), black, 1)
    cv2.line(img, (right_x - 5, 45), (right_x + 5, 45), black, 1)
    cv2.line(img, (right_x - 5, 785), (right_x + 5, 785), black, 1)

    # a legitimate thin partition: 5 px thick, 150 px long, free-standing inside a room
    part = (700, 640, 850, 645)  # x1, y1, x2, y2 at working size
    cv2.rectangle(img, (part[0], part[1]), (part[2] - 1, part[3] - 1), black, -1)

    gt = ground_truth(width, height)
    partition = np.zeros((height, width), np.uint8)
    cv2.rectangle(partition, (part[0], part[1]), (part[2] - 1, part[3] - 1), 255, -1)
    dim_bands = np.zeros((height, width), np.uint8)
    dim_bands[top_y - 6: top_y + 7, 40:1166] = 255
    dim_bands[40:791, right_x - 6: right_x + 7] = 255
    truth = {
        "wall": cv2.bitwise_or(gt["wall"], partition),
        "partition": partition,
        "footprint": gt["footprint"],
        "furniture": gt["furniture"],
        "furniture_outline": gt["furniture_outline"],
        "dimension": dim_bands,
    }
    return img, truth


if __name__ == "__main__":
    for p in generate_samples():
        print("wrote", p)
