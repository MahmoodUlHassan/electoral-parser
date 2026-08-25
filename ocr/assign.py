"""Page-level OCR → assign tokens to card boxes (experiment path)."""

from __future__ import annotations

from parser.config import DEFAULT_LAYOUT, LayoutProfile
from parser.models import Box, CardDetection, OcrToken


def adaptive_ocr_scale(card_height_px: float) -> float:
    """Scale factor for page/row OCR from typical card height on the raster."""
    h = float(card_height_px)
    if h > 180:
        return 1.0
    if h > 150:
        return 1.5
    if h > 120:
        return 2.0
    return 2.0


def scale_token_to_page(token: OcrToken, scale: float) -> OcrToken:
    if scale == 1.0:
        return token
    inv = 1.0 / scale
    return OcrToken(
        text=token.text,
        confidence=token.confidence,
        bbox=[[p[0] * inv, p[1] * inv] for p in token.bbox],
    )


def _overlap_area(token: OcrToken, box: Box) -> float:
    xs = [p[0] for p in token.bbox]
    ys = [p[1] for p in token.bbox]
    tx0, tx1 = min(xs), max(xs)
    ty0, ty1 = min(ys), max(ys)
    ix0 = max(tx0, box.x)
    iy0 = max(ty0, box.y)
    ix1 = min(tx1, box.x2)
    iy1 = min(ty1, box.y2)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return float((ix1 - ix0) * (iy1 - iy0))


def _token_area(token: OcrToken) -> float:
    xs = [p[0] for p in token.bbox]
    ys = [p[1] for p in token.bbox]
    return max(1.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))


def _in_photo_strip(token: OcrToken, card: CardDetection, layout: LayoutProfile) -> bool:
    """True if token centroid falls in the card's photo column (skip assignment noise)."""
    b = card.box
    pr = layout.photo_region
    px0 = b.x + pr.x * b.w
    px1 = b.x + (pr.x + pr.w) * b.w
    py0 = b.y + pr.y * b.h
    py1 = b.y + (pr.y + pr.h) * b.h
    return px0 <= token.cx <= px1 and py0 <= token.cy <= py1


def assign_tokens_to_cards(
    tokens: list[OcrToken],
    cards: list[CardDetection],
    layout: LayoutProfile = DEFAULT_LAYOUT,
    *,
    min_overlap_frac: float = 0.35,
) -> dict[int, list[OcrToken]]:
    """Map page-space tokens → card index via max bbox overlap (centroid fallback)."""
    occupied = [c for c in cards if c.occupied]
    buckets: dict[int, list[OcrToken]] = {c.index: [] for c in occupied}
    if not occupied or not tokens:
        return buckets

    for token in tokens:
        best_i: int | None = None
        best_score = 0.0
        for card in occupied:
            if _in_photo_strip(token, card, layout):
                # Keep EPIC strip (top of photo gutter); skip body "Photo Available".
                if token.cy > card.box.y + 0.22 * card.box.h:
                    continue
            overlap = _overlap_area(token, card.box)
            frac = overlap / _token_area(token)
            if frac > best_score:
                best_score = frac
                best_i = card.index
        if best_i is None or best_score < min_overlap_frac:
            # Centroid-in-box fallback for thin tokens.
            for card in occupied:
                b = card.box
                if b.x <= token.cx <= b.x2 and b.y <= token.cy <= b.y2:
                    if _in_photo_strip(token, card, layout) and token.cy > b.y + 0.22 * b.h:
                        continue
                    best_i = card.index
                    break
            else:
                continue
        buckets[best_i].append(token)
    return buckets


def tokens_to_card_local(tokens: list[OcrToken], card: CardDetection) -> list[OcrToken]:
    """Shift page-space token bboxes into card-local coordinates for parse_voter_card."""
    ox, oy = card.box.x, card.box.y
    local: list[OcrToken] = []
    for token in tokens:
        local.append(
            OcrToken(
                text=token.text,
                confidence=token.confidence,
                bbox=[[p[0] - ox, p[1] - oy] for p in token.bbox],
            )
        )
    return local
