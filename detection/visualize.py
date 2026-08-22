from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from parser.config import DEFAULT_LAYOUT, LayoutProfile, RelativeBox
from parser.models import CardDetection


def draw_detections(image_bgr: np.ndarray, cards: list[CardDetection]) -> np.ndarray:
    vis = image_bgr.copy()
    for card in cards:
        color = (0, 180, 0) if card.occupied else (80, 80, 80)
        x, y, w, h = card.box.x, card.box.y, card.box.w, card.box.h
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        label = str(card.index + 1)
        cv2.putText(
            vis,
            label,
            (x + 8, y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 220),
            2,
            cv2.LINE_AA,
        )
    return vis


def crop_card(image_bgr: np.ndarray, card: CardDetection) -> np.ndarray:
    b = card.box
    return image_bgr[b.y : b.y2, b.x : b.x2].copy()


def crop_card_padded(
    image_bgr: np.ndarray,
    card: CardDetection,
    *,
    pad_top: float = 0.10,
    pad_bottom: float = 0.14,
    pad_x: float = 0.02,
) -> np.ndarray:
    """Widen the grid cell slightly — last pages often clip serial/EPIC/Age."""
    h, w = image_bgr.shape[:2]
    b = card.box
    top = max(0, b.y - int(b.h * pad_top))
    bottom = min(h, b.y2 + int(b.h * pad_bottom))
    left = max(0, b.x - int(b.w * pad_x))
    right = min(w, b.x2 + int(b.w * pad_x))
    return image_bgr[top:bottom, left:right].copy()


def crop_relative(image_bgr: np.ndarray, region: RelativeBox) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x, y, bw, bh = region.pixel_box(w, h)
    if bw <= 0 or bh <= 0:
        return image_bgr[0:0, 0:0].copy()
    return image_bgr[y : y + bh, x : x + bw].copy()


def _paste_region(dst: np.ndarray, src: np.ndarray, region: RelativeBox) -> None:
    h, w = src.shape[:2]
    x, y, bw, bh = region.pixel_box(w, h)
    if bw <= 0 or bh <= 0:
        return
    dst[y : y + bh, x : x + bw] = src[y : y + bh, x : x + bw]


def crop_card_for_ocr(
    image_bgr: np.ndarray,
    card: CardDetection,
    layout: LayoutProfile = DEFAULT_LAYOUT,
) -> np.ndarray:
    """Full-card-sized crop with photo wiped; text_region + epic strips kept."""
    return prepare_card_ocr_image(crop_card(image_bgr, card), layout)


def prepare_card_ocr_image(
    crop: np.ndarray,
    layout: LayoutProfile = DEFAULT_LAYOUT,
) -> np.ndarray:
    """Wipe photo body on an already-cropped card; keep text + epic strips."""
    out = np.full_like(crop, 255)
    _paste_region(out, crop, layout.text_region)
    _paste_region(out, crop, layout.epic_region)
    _paste_region(out, crop, layout.epic_fallback_region)
    return out


def save_card_crop(image_bgr: np.ndarray, card: CardDetection, path: Path) -> np.ndarray:
    crop = crop_card(image_bgr, card)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)
    return crop
