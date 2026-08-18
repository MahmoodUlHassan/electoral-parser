from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

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


def crop_relative(image_bgr: np.ndarray, region) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x, y, bw, bh = region.pixel_box(w, h)
    if bw <= 0 or bh <= 0:
        return image_bgr[0:0, 0:0].copy()
    return image_bgr[y : y + bh, x : x + bw].copy()


def save_card_crop(image_bgr: np.ndarray, card: CardDetection, path: Path) -> np.ndarray:
    crop = crop_card(image_bgr, card)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)
    return crop
