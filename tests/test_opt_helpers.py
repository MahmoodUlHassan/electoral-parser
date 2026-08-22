from __future__ import annotations

import numpy as np

from detection.visualize import crop_card_for_ocr
from main import parse_success
from parser.config import DEFAULT_LAYOUT
from parser.models import Box, CardDetection, DetectionMethod, VoterRecord
from parser.pipeline import needs_band_refill


def test_crop_card_for_ocr_wipes_photo_keeps_text():
    h, w = 100, 200
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    pr = DEFAULT_LAYOUT.photo_region.pixel_box(w, h)
    image[pr[1] : pr[1] + pr[3], pr[0] : pr[0] + pr[2]] = (0, 0, 255)
    card = CardDetection(
        index=0,
        box=Box(0, 0, w, h),
        occupied=True,
        method=DetectionMethod.GRID,
    )
    out = crop_card_for_ocr(image, card)
    assert out.shape == image.shape
    cy = int(h * 0.55)
    cx = int(w * 0.88)
    assert tuple(int(v) for v in out[cy, cx]) == (255, 255, 255)
    assert tuple(int(v) for v in out[50, 20]) == (10, 20, 30)


def test_needs_band_refill_skips_complete():
    rec = VoterRecord(
        serial_no=1,
        epic="ABC1234567",
        name="A",
        relation_type="Father",
        relative_name="B",
        house_no="1",
        age=30,
        gender="M",
        page=3,
        part_no=1,
        constituency="X",
        section="Y",
    )
    assert needs_band_refill(rec) is False
    rec.house_no = None
    assert needs_band_refill(rec) is True
    rec.house_no = "1"
    rec.epic = None
    assert needs_band_refill(rec) is True


def test_age_crowded_region_configured():
    assert DEFAULT_LAYOUT.age_crowded_region.y >= 0.75
    assert DEFAULT_LAYOUT.age_crowded_region.y + DEFAULT_LAYOUT.age_crowded_region.h <= 1.01


def test_parse_success_threshold():
    assert parse_success({"validCount": 90, "extractedTotal": 100, "expectedTotal": 100})
    assert not parse_success({"validCount": 80, "extractedTotal": 100, "expectedTotal": 100})
    assert parse_success({"validCount": 9, "extractedTotal": 10, "expectedTotal": None})
