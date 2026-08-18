from __future__ import annotations

import logging

import numpy as np

from detection.contours import detect_cards_contours
from detection.geometry import ink_ratio, line_peaks
from detection.grid import detect_cards_grid
from parser.config import DEFAULT_LAYOUT, LayoutProfile
from parser.models import DetectionMethod, PageType

logger = logging.getLogger("electoral.detection.classifier")


def classify_page(image_bgr: np.ndarray, layout: LayoutProfile = DEFAULT_LAYOUT) -> PageType:
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    h, w = gray.shape[:2]
    binary = (gray < 120).astype(np.uint8)
    h_lines = line_peaks(binary.sum(axis=1), min_val=w * 0.40, min_gap=4)
    v_strong = line_peaks(binary.sum(axis=0), min_val=h * 0.18, min_gap=6)
    ink = ink_ratio(gray)

    if ink < 0.004:
        return PageType.BLANK

    cards = detect_cards_contours(image_bgr, layout)
    if len(cards) >= layout.min_occupied_to_keep:
        return PageType.VOTER

    # Full pages with faint borders: line signature without card contours.
    if len(v_strong) >= 8 and len(h_lines) >= 16:
        occupied = [c for c in detect_cards_grid(gray, layout) if c.occupied]
        if len(occupied) >= layout.min_full_page_cards:
            return PageType.VOTER

    if len(h_lines) >= 10:
        return PageType.HEADER
    if len(h_lines) <= 3:
        return PageType.MAP
    return PageType.SUPPLEMENT


def detect_cards(
    image_bgr: np.ndarray,
    layout: LayoutProfile = DEFAULT_LAYOUT,
) -> tuple[list, DetectionMethod]:
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    grid = detect_cards_grid(gray, layout)
    occupied = [c for c in grid if c.occupied]

    if len(occupied) >= layout.min_full_page_cards:
        return occupied, DetectionMethod.GRID
    if 1 <= len(occupied) < layout.min_full_page_cards:
        contours = detect_cards_contours(image_bgr, layout)
        if len(contours) > len(occupied) + 3:
            logger.info(
                "grid found %d cards, contour found %d — using contour fallback",
                len(occupied),
                len(contours),
            )
            return contours, DetectionMethod.CONTOUR
        return occupied, DetectionMethod.GRID

    contours = detect_cards_contours(image_bgr, layout)
    logger.info("grid empty, contour fallback found %d cards", len(contours))
    return contours, DetectionMethod.CONTOUR
