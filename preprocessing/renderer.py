from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pymupdf

from parser.config import DEFAULT_DPI, NATIVE_EXTRACT

logger = logging.getLogger("electoral.preprocess")


def _pixmap_to_bgr(pix: pymupdf.Pixmap) -> np.ndarray:
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        return np.repeat(arr, 3, axis=2)
    # RGB → BGR for OpenCV
    return arr[:, :, ::-1].copy()


def _decode_embedded(doc: pymupdf.Document, page: pymupdf.Page) -> np.ndarray | None:
    images = page.get_images(full=True)
    if len(images) != 1:
        return None
    xref = images[0][0]
    info = doc.extract_image(xref)
    if not info:
        return None
    import cv2

    buf = np.frombuffer(info["image"], dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    # Native extract is only a win when the image fills the page.
    w, h = page.rect.width, page.rect.height
    if abs(info["width"] - w) < 3 and abs(info["height"] - h) < 3:
        return img
    return None


def render_page(
    doc: pymupdf.Document,
    page_index: int,
    *,
    dpi: int = DEFAULT_DPI,
    scale: float | None = None,
    prefer_native: bool = NATIVE_EXTRACT,
) -> np.ndarray:
    """Render a PDF page to a BGR numpy array.

    Default: extract the full-page JPEG (true pixels). `--scale 4` matches
    `page.get_pixmap(matrix=fitz.Matrix(4,4))`. `--dpi 300` uses Matrix(dpi/72).
    """
    page = doc[page_index]
    if prefer_native and scale is None:
        native = _decode_embedded(doc, page)
        if native is not None:
            return native

    if scale is None:
        scale = dpi / 72.0
    matrix = pymupdf.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return _pixmap_to_bgr(pix)


def save_debug_image(image: np.ndarray, path: Path) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
