from __future__ import annotations

import logging

import numpy as np

from detection.geometry import cluster_positions, ink_ratio, line_peaks
from parser.config import DEFAULT_LAYOUT, LayoutProfile
from parser.models import Box, CardDetection, DetectionMethod

logger = logging.getLogger("electoral.detection.grid")

_INSET = 6


def _binarize(gray: np.ndarray) -> np.ndarray:
    return (gray < 120).astype(np.uint8)


def detect_content_frame(gray: np.ndarray, layout: LayoutProfile = DEFAULT_LAYOUT) -> Box:
    h, w = gray.shape[:2]
    binary = _binarize(gray)
    v_lines = line_peaks(binary.sum(axis=0), min_val=h * 0.12, min_gap=6)
    h_lines = line_peaks(binary.sum(axis=1), min_val=w * 0.40, min_gap=4)

    left_candidates = [x for x in v_lines if x < w * 0.08]
    right_candidates = [x for x in v_lines if x > w * 0.92]
    left = min(left_candidates) if left_candidates else int(layout.content_left * w)
    right = max(right_candidates) if right_candidates else int(layout.content_right * w)

    top_candidates = [y for y in h_lines if h * 0.015 < y < h * 0.12]
    bottom_candidates = [y for y in h_lines if y > h * 0.85]
    top = min(top_candidates) if top_candidates else int(layout.content_top * h)
    bottom = max(bottom_candidates) if bottom_candidates else int(layout.content_bottom * h)
    if bottom <= top + 50:
        top, bottom = int(layout.content_top * h), int(layout.content_bottom * h)
    return Box(left, top, max(1, right - left), max(1, bottom - top))


def _row_dividers(gray: np.ndarray, frame: Box, layout: LayoutProfile) -> list[int]:
    h, w = gray.shape[:2]
    h_lines = line_peaks(_binarize(gray).sum(axis=1), min_val=w * 0.40, min_gap=4)
    inner = [y for y in h_lines if frame.y - 8 <= y <= frame.y2 + 8]
    clustered = cluster_positions(inner, gap=20)
    if len(clustered) >= layout.rows + 1:
        return clustered[: layout.rows + 1]
    pitch = frame.h / layout.rows
    return [int(round(frame.y + i * pitch)) for i in range(layout.rows + 1)]


def detect_cards_grid(
    gray: np.ndarray,
    layout: LayoutProfile = DEFAULT_LAYOUT,
) -> list[CardDetection]:
    """Divide the printable table into a 3×N grid and keep occupied cells.

    N is `layout.rows` (10). Short section-tail pages simply yield empty cells.
    """
    h, w = gray.shape[:2]
    frame = detect_content_frame(gray, layout)
    col_w = frame.w / layout.columns
    y_splits = _row_dividers(gray, frame, layout)

    cards: list[CardDetection] = []
    index = 0
    for row in range(layout.rows):
        y0 = y_splits[row] + _INSET
        y1 = y_splits[row + 1] - _INSET
        for col in range(layout.columns):
            x0 = int(round(frame.x + col * col_w)) + _INSET
            x1 = int(round(frame.x + (col + 1) * col_w)) - _INSET
            box = Box(x0, y0, max(1, x1 - x0), max(1, y1 - y0)).clip(w, h)
            crop = gray[box.y : box.y2, box.x : box.x2]
            margin = 10
            probe = (
                crop[margin:-margin, margin:-margin]
                if crop.shape[0] > 2 * margin and crop.shape[1] > 2 * margin
                else crop
            )
            occupied = ink_ratio(probe) >= layout.occupancy_ink_ratio
            cards.append(
                CardDetection(index=index, box=box, occupied=occupied, method=DetectionMethod.GRID)
            )
            index += 1
    return cards
