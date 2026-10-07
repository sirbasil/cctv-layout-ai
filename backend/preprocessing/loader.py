"""Load PNG / JPG / JPEG / PDF into ONE consistent type: an OpenCV BGR uint8 array.

Everything downstream only ever sees `np.ndarray, shape (H, W, 3), dtype uint8, BGR`,
so the rest of the pipeline never needs to care what the original file was.
"""
from pathlib import Path

import cv2
import numpy as np
import pymupdf  # PyMuPDF
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".png": "png", ".jpg": "jpg", ".jpeg": "jpg"}
PDF_EXTENSIONS = {".pdf": "pdf"}
SUPPORTED_EXTENSIONS = {**IMAGE_EXTENSIONS, **PDF_EXTENSIONS}

# PDFs are vector/page-sized, so we choose a render resolution. 150 DPI is plenty for
# wall detection; the cap stops a huge sheet (e.g. an A0 drawing) from eating all RAM.
PDF_RENDER_DPI = 150
PDF_MAX_RENDER_PIXELS = 4000  # longest side, before normalize_image shrinks it further


class LayoutLoadError(ValueError):
    """Base class for all loading problems (easy to catch in one place)."""


class UnsupportedFormatError(LayoutLoadError):
    """File extension is not one of PNG / JPG / JPEG / PDF."""


class InvalidFileError(LayoutLoadError):
    """File has a supported extension but is missing, empty, corrupt or unreadable."""


def get_source_format(path) -> str:
    """Return the canonical format name: 'png', 'jpg' or 'pdf' (jpeg -> 'jpg')."""
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported file format '{ext or '(no extension)'}' for '{path}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return SUPPORTED_EXTENSIONS[ext]


def load_layout(path) -> np.ndarray:
    """Load a floor plan file and return an OpenCV BGR uint8 array (H, W, 3).

    PDFs: only the FIRST page is used.
    Raises UnsupportedFormatError / InvalidFileError with a readable message.
    """
    path = Path(path)
    fmt = get_source_format(path)  # raises for unsupported extensions
    if not path.is_file():
        raise InvalidFileError(f"File not found: '{path}'")
    if path.stat().st_size == 0:
        raise InvalidFileError(f"File is empty: '{path}'")

    if fmt == "pdf":
        return _load_pdf(path)
    return _load_raster(path)


# --------------------------------------------------------------------------- helpers
def _load_raster(path: Path) -> np.ndarray:
    # np.fromfile + imdecode (instead of cv2.imread) so non-ASCII paths work on Windows.
    data = np.fromfile(str(path), dtype=np.uint8)
    try:
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)  # UNCHANGED keeps alpha if present
    except cv2.error:  # e.g. absurdly large dimensions; let Pillow try / give a clear error
        img = None
    if img is None:
        return _load_raster_with_pillow(path)  # rare variants OpenCV can't decode
    return _to_bgr_uint8(img)


def _load_raster_with_pillow(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as im:
            rgba = im.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(white, rgba).convert("RGB")
        return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidFileError(f"Could not decode image '{path}': {exc}") from exc


def _to_bgr_uint8(img: np.ndarray) -> np.ndarray:
    """Force any decoded image into (H, W, 3) uint8 BGR."""
    if img.dtype == np.uint16:  # 16-bit PNGs
        img = (img / 257).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if img.ndim == 2:  # grayscale
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:  # BGRA -> composite onto WHITE (transparent bg must not become black walls)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        bgr = img[:, :, :3].astype(np.float32)
        return (bgr * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return np.ascontiguousarray(img[:, :, :3])


def _load_pdf(path: Path) -> np.ndarray:
    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:  # PyMuPDF raises several different error types
        raise InvalidFileError(f"Could not open PDF '{path}': {exc}") from exc

    with doc:
        if doc.needs_pass:
            raise InvalidFileError(f"PDF is password protected: '{path}'")
        if doc.page_count == 0:
            raise InvalidFileError(f"PDF has no pages: '{path}'")
        try:
            page = doc.load_page(0)
            zoom = PDF_RENDER_DPI / 72.0  # PDF user space is 72 units per inch
            longest = max(page.rect.width, page.rect.height) * zoom
            if longest > PDF_MAX_RENDER_PIXELS:
                zoom *= PDF_MAX_RENDER_PIXELS / longest
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(zoom, zoom),
                colorspace=pymupdf.csRGB,
                alpha=False,  # white background, no transparency
            )
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            rgb = rgb.copy()  # detach from the pixmap buffer before the doc closes
        except Exception as exc:
            raise InvalidFileError(f"Could not render first page of '{path}': {exc}") from exc

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
