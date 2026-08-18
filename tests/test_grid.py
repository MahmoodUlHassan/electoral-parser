import numpy as np

from detection.grid import detect_cards_grid
from parser.config import LayoutProfile


def _fake_voter_page(w=1983, h=2806) -> np.ndarray:
    """White page with a 3×10 grid of dark card borders matching ECI pitch."""
    img = np.full((h, w), 255, dtype=np.uint8)
    left, right = int(0.0237 * w), int(0.9743 * w)
    top, bottom = int(0.0335 * h), int(0.9701 * h)
    cols, rows = 3, 10
    col_w = (right - left) / cols
    row_h = (bottom - top) / rows
    for r in range(rows + 1):
        y = int(top + r * row_h)
        img[max(0, y - 1) : y + 2, left:right] = 0
    for c in range(cols + 1):
        x = int(left + c * col_w)
        img[top:bottom, max(0, x - 1) : x + 2] = 0
    for r in range(rows):
        for c in range(cols):
            x0 = int(left + c * col_w) + 15
            y0 = int(top + r * row_h) + 15
            x1 = int(left + (c + 1) * col_w) - 15
            y1 = int(top + (r + 1) * row_h) - 15
            img[y0:y1, x0:x1] = 40
    return img


def test_grid_finds_thirty_occupied_cards():
    gray = _fake_voter_page()
    cards = detect_cards_grid(gray, LayoutProfile())
    occupied = [c for c in cards if c.occupied]
    assert len(cards) == 30
    assert len(occupied) == 30
    xs = sorted({c.box.x for c in occupied})
    assert len(xs) == 3
