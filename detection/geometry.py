from __future__ import annotations

import numpy as np

from parser.config import DEFAULT_LAYOUT, LayoutProfile
from parser.models import Box


def cluster_positions(values: list[int] | np.ndarray, gap: int) -> list[int]:
    if len(values) == 0:
        return []
    ordered = sorted(int(v) for v in values)
    groups: list[list[int]] = [[ordered[0]]]
    for v in ordered[1:]:
        if v - groups[-1][-1] > gap:
            groups.append([v])
        else:
            groups[-1].append(v)
    return [int(np.median(g)) for g in groups]


def line_peaks(projection: np.ndarray, min_val: float, min_gap: int = 4) -> list[int]:
    idx = np.where(projection >= min_val)[0]
    if len(idx) == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > min_gap:
            groups.append((start, prev))
            start = i
        prev = i
    groups.append((start, prev))
    return [int((a + b) / 2) for a, b in groups]


def ink_ratio(gray: np.ndarray, thresh: int = 140) -> float:
    if gray.size == 0:
        return 0.0
    return float(np.mean(gray < thresh))


def is_eci_card_box(
    box: Box, page_w: int, page_h: int, layout: LayoutProfile = DEFAULT_LAYOUT
) -> bool:
    """True for the printed 3×10 voter boxes, not cover/summary tables."""
    aspect = box.w / max(box.h, 1)
    if abs(aspect - layout.card_aspect) > 0.35:
        return False
    w_frac = box.w / max(page_w, 1)
    h_frac = box.h / max(page_h, 1)
    return 0.26 <= w_frac <= 0.36 and 0.070 <= h_frac <= 0.12
