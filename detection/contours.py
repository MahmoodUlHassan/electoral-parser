from __future__ import annotations

import logging

import cv2
import numpy as np

from detection.geometry import is_eci_card_box
from parser.config import DEFAULT_LAYOUT, LayoutProfile
from parser.models import Box, CardDetection, DetectionMethod

logger = logging.getLogger("electoral.detection.contours")


def detect_cards_contours(
    image_bgr: np.ndarray,
    layout: LayoutProfile = DEFAULT_LAYOUT,
) -> list[CardDetection]:
    """Adaptive-threshold + morphology + findContours fallback."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    h, w = gray.shape[:2]
    page_area = h * w
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
    closed = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 0.012 * page_area
    max_area = 0.07 * page_area
    raw: list[Box] = []
    for contour in contours:
        x, y, rw, rh = cv2.boundingRect(contour)
        area = rw * rh
        if area < min_area or area > max_area:
            continue
        aspect = rw / max(rh, 1)
        if not (1.6 <= aspect <= 3.6):
            continue
        if rh < 70 or rw < 180:
            continue
        raw.append(Box(x, y, rw, rh))

    # Dedup overlapping detections (border drawn twice).
    raw.sort(key=lambda b: b.area, reverse=True)
    unique: list[Box] = []
    for box in raw:
        if any(abs(box.x - u.x) < 18 and abs(box.y - u.y) < 18 for u in unique):
            continue
        unique.append(box)

    unique.sort(key=lambda b: (b.y // 20, b.x))
    unique = [b for b in unique if is_eci_card_box(b, w, h, layout)]
    cards = [
        CardDetection(index=i, box=b, occupied=True, method=DetectionMethod.CONTOUR)
        for i, b in enumerate(unique)
    ]
    logger.debug("contour detector found %d cards", len(cards))
    return cards
