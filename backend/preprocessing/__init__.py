"""Floor-plan preprocessing: any supported input file -> one canonical layout."""
from .loader import (
    InvalidFileError,
    LayoutLoadError,
    UnsupportedFormatError,
    get_source_format,
    load_layout,
)
from .footprint import build_building_mask, get_interior_free_mask
from .normalize import normalize_image
from .pipeline import preprocess
from .walls import extract_wall_mask, get_free_space

__all__ = [
    "InvalidFileError",
    "LayoutLoadError",
    "UnsupportedFormatError",
    "build_building_mask",
    "extract_wall_mask",
    "get_free_space",
    "get_interior_free_mask",
    "get_source_format",
    "load_layout",
    "normalize_image",
    "preprocess",
]
