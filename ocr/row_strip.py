"""Row-strip OCR: 10 horizontal strips → wipe photo/borders → OCR → assign by x."""

from __future__ import annotations

import numpy as np

from detection.visualize import crop_relative
from ocr.assign import adaptive_ocr_scale
from parser.config import DEFAULT_LAYOUT, LayoutProfile
from parser.models import Box, CardDetection, OcrToken


def group_cards_into_rows(
    cards: list[CardDetection],
    layout: LayoutProfile = DEFAULT_LAYOUT,
) -> list[list[CardDetection]]:
    """Partition grid cards into layout.rows lists (may be empty / sparse)."""
    by_index = {c.index: c for c in cards if c.occupied}
    rows: list[list[CardDetection]] = []
    for r in range(layout.rows):
        row: list[CardDetection] = []
        for c in range(layout.columns):
            idx = r * layout.columns + c
            if idx in by_index:
                row.append(by_index[idx])
        if row:
            rows.append(row)
    return rows


def row_union_box(row_cards: list[CardDetection]) -> Box:
    x0 = min(c.box.x for c in row_cards)
    y0 = min(c.box.y for c in row_cards)
    x1 = max(c.box.x2 for c in row_cards)
    y1 = max(c.box.y2 for c in row_cards)
    return Box(x0, y0, x1 - x0, y1 - y0)


def _fill_white(img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    h_img, w_img = img.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = 255


def prepare_row_strip(
    page_bgr: np.ndarray,
    row_cards: list[CardDetection],
    layout: LayoutProfile = DEFAULT_LAYOUT,
    *,
    border_px: int = 3,
) -> tuple[np.ndarray, Box]:
    """Crop the row union; wipe photo bodies (keep EPIC) and card borders/gutters."""
    union = row_union_box(row_cards)
    strip = page_bgr[union.y : union.y2, union.x : union.x2].copy()
    # Start white, paste text+epic per card (same idea as prepare_card_ocr_image).
    out = np.full_like(strip, 255)
    for card in row_cards:
        lx = card.box.x - union.x
        ly = card.box.y - union.y
        cw, ch = card.box.w, card.box.h
        card_crop = page_bgr[card.box.y : card.box.y2, card.box.x : card.box.x2]
        prepared = np.full_like(card_crop, 255)
        for region in (layout.text_region, layout.epic_region, layout.epic_fallback_region):
            rx, ry, rw, rh = region.pixel_box(cw, ch)
            if rw <= 0 or rh <= 0:
                continue
            prepared[ry : ry + rh, rx : rx + rw] = card_crop[ry : ry + rh, rx : rx + rw]
        # Erase inner border ring so grid lines don't confuse det.
        if border_px > 0:
            prepared[:border_px, :] = 255
            prepared[-border_px:, :] = 255
            prepared[:, :border_px] = 255
            prepared[:, -border_px:] = 255
        out[ly : ly + ch, lx : lx + cw] = prepared
    return out, union


def assign_tokens_by_x(
    tokens: list[OcrToken],
    row_cards: list[CardDetection],
    union: Box,
) -> dict[int, list[OcrToken]]:
    """Assign strip-local tokens to cards by centroid x (column bands)."""
    buckets: dict[int, list[OcrToken]] = {c.index: [] for c in row_cards}
    if not tokens or not row_cards:
        return buckets
    # Strip-local card x ranges
    local = [(c, c.box.x - union.x, c.box.x2 - union.x) for c in row_cards]
    for token in tokens:
        cx = token.cx
        best = None
        best_dist = 1e18
        for card, x0, x1 in local:
            if x0 <= cx <= x1:
                best = card.index
                break
            mid = 0.5 * (x0 + x1)
            dist = abs(cx - mid)
            if dist < best_dist:
                best_dist = dist
                best = card.index
        if best is not None:
            buckets[best].append(token)
    return buckets


def tokens_to_card_local_from_strip(
    tokens: list[OcrToken],
    card: CardDetection,
    union: Box,
) -> list[OcrToken]:
    """Map strip-local token bboxes into card-local coordinates."""
    ox = card.box.x - union.x
    oy = card.box.y - union.y
    return [
        OcrToken(
            text=t.text,
            confidence=t.confidence,
            bbox=[[p[0] - ox, p[1] - oy] for p in t.bbox],
        )
        for t in tokens
    ]


def row_ocr_scale(row_cards: list[CardDetection]) -> float:
    if not row_cards:
        return 2.0
    h = float(np.median([c.box.h for c in row_cards]))
    return adaptive_ocr_scale(h)
