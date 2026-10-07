"""Size normalization so every downstream stage works at a predictable resolution."""
import cv2
import numpy as np


def normalize_image(image: np.ndarray, max_dimension: int = 1200) -> np.ndarray:
    """Downscale so the longest side is <= max_dimension. Aspect ratio is preserved.

    - Only ever shrinks. Smaller images are returned unchanged (upscaling adds no
      information and would just make later stages slower).
    - Uses INTER_AREA, the right interpolation for shrinking (keeps thin wall lines
      from vanishing the way nearest-neighbour would).
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("normalize_image expects a BGR image of shape (H, W, 3)")
    if max_dimension < 1:
        raise ValueError("max_dimension must be >= 1")

    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dimension:
        return image

    scale = max_dimension / longest
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
