from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RelativeBox:
    """Axis-aligned region as fractions of a parent box (0-1)."""

    x: float
    y: float
    w: float
    h: float

    def pixel_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        x = int(round(self.x * width))
        y = int(round(self.y * height))
        w = max(1, int(round(self.w * width)))
        h = max(1, int(round(self.h * height)))
        x = max(0, min(max(0, width - 1), x))
        y = max(0, min(max(0, height - 1), y))
        w = min(w, width - x)
        h = min(h, height - y)
        return x, y, w, h


@dataclass(frozen=True, slots=True)
class LayoutProfile:
    """Fixed ECI 2026 English photo-roll layout.

    Measured on Telangana S29 English SIR drafts (1983×2806 rasters):
    3 columns × 10 rows, ~14 px gutters, photo strip on the right ~25%.
    New state/year layouts become additional profiles, not if/else sprawl.
    """

    name: str = "eci-2026-en-3x10"
    columns: int = 3
    rows: int = 10
    min_full_page_cards: int = 25
    min_occupied_to_keep: int = 1
    card_aspect: float = 2.47
    # Relative to full page. Used when line detection is noisy.
    content_left: float = 0.0237
    content_right: float = 0.9743
    content_top: float = 0.0335
    content_bottom: float = 0.9701
    header_max: float = 0.035
    footer_min: float = 0.968
    # Inside a voter card (fractions of the card crop).
    serial_box: RelativeBox = RelativeBox(0.00, 0.00, 0.16, 0.28)
    # Short top strip into the photo gutter — tall enough for EPIC, short enough to
    # avoid "Photo Available" which makes tiny OCR drop the last digit.
    epic_region: RelativeBox = RelativeBox(0.50, 0.00, 0.48, 0.18)
    # Fallback when primary strip still truncates (measured: x>0.55, y<0.28).
    epic_fallback_region: RelativeBox = RelativeBox(0.55, 0.00, 0.44, 0.28)
    photo_region: RelativeBox = RelativeBox(0.76, 0.02, 0.24, 0.96)
    # Body + serial column; EPIC pasted separately for OCR crops.
    text_region: RelativeBox = RelativeBox(0.01, 0.00, 0.74, 0.99)
    # Ink bands below this y-fraction are body lines (Name…Age), not serial/EPIC.
    body_line_min_y: float = 0.18
    # Fallback only if projection finds too few lines (4-line layout).
    house_region: RelativeBox = RelativeBox(0.01, 0.415, 0.74, 0.125)
    age_region: RelativeBox = RelativeBox(0.01, 0.530, 0.74, 0.145)
    age_fallback_region: RelativeBox = RelativeBox(0.01, 0.640, 0.74, 0.16)
    # 6-line wraps push Age to the bottom edge (or just below fixed age bands).
    age_crowded_region: RelativeBox = RelativeBox(0.01, 0.780, 0.74, 0.20)
    occupancy_ink_ratio: float = 0.03
    epic_pattern: str = r"^[A-Z]{3}[0-9]{7}$"
    age_min: int = 18
    age_max: int = 120


DEFAULT_LAYOUT = LayoutProfile()
DEFAULT_DPI = 300
# These PDFs store 1 image pixel = 1 PDF point (~240 DPI A4). Native extract
# is sharper than Matrix(4,4) which only upsamples JPEG. Scale 4 remains available.
DEFAULT_RENDER_SCALE = 1.0
NATIVE_EXTRACT = True
