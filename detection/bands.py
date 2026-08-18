from __future__ import annotations

import cv2
import numpy as np

from parser.config import DEFAULT_LAYOUT, LayoutProfile, RelativeBox


def detect_body_line_regions(
    image_bgr: np.ndarray,
    layout: LayoutProfile = DEFAULT_LAYOUT,
) -> list[RelativeBox]:
    """Horizontal ink bands in the card text column, skipping the serial/EPIC row.

    4-line cards: Name, Relation, House, Age.
    5-line: one of Name/Relation wraps.
    6-line: both wrap.
    """
    if image_bgr.size == 0:
        return []
    h, w = image_bgr.shape[:2]
    x, y, bw, bh = layout.text_region.pixel_box(w, h)
    roi = image_bgr[y : y + bh, x : x + bw]
    if roi.size == 0:
        return []
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    ink = gray < 90
    ink[:3, :] = False
    ink[-3:, :] = False
    ink[:, :3] = False
    ink[:, -3:] = False
    row = ink.mean(axis=1)
    on = row > 0.012
    raw: list[tuple[int, int]] = []
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if j - i >= 6:
                raw.append((y + i, y + j))
            i = j
        else:
            i += 1
    body: list[tuple[int, int]] = []
    min_y = int(layout.body_line_min_y * h)
    for y0, y1 in raw:
        if (y0 + y1) / 2 < min_y:
            continue
        body.append((y0, y1))
    if not body:
        return []
    boxes: list[RelativeBox] = []
    pad = max(3, int(0.012 * h))
    for idx, (y0, y1) in enumerate(body):
        prev_end = body[idx - 1][1] if idx else min_y
        next_start = body[idx + 1][0] if idx + 1 < len(body) else h - 2
        py0 = max(prev_end, y0 - pad)
        py1 = min(next_start, y1 + pad)
        if py1 <= py0:
            py0, py1 = y0, y1
        boxes.append(
            RelativeBox(
                x=layout.text_region.x,
                y=py0 / h,
                w=layout.text_region.w,
                h=max(1, py1 - py0) / h,
            )
        )
    return boxes
