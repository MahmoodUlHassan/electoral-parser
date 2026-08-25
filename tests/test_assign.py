from __future__ import annotations

from ocr.assign import adaptive_ocr_scale, assign_tokens_to_cards
from parser.models import Box, CardDetection, DetectionMethod, OcrToken


def test_adaptive_ocr_scale():
    assert adaptive_ocr_scale(200) == 1.0
    assert adaptive_ocr_scale(160) == 1.5
    assert adaptive_ocr_scale(130) == 2.0
    assert adaptive_ocr_scale(100) == 2.0


def test_assign_tokens_by_overlap():
    cards = [
        CardDetection(0, Box(0, 0, 100, 100), True, DetectionMethod.GRID),
        CardDetection(1, Box(120, 0, 100, 100), True, DetectionMethod.GRID),
    ]
    tokens = [
        OcrToken("A", 0.9, [[10, 10], [40, 10], [40, 30], [10, 30]]),
        OcrToken("B", 0.9, [[130, 10], [160, 10], [160, 30], [130, 30]]),
    ]
    buckets = assign_tokens_to_cards(tokens, cards)
    assert [t.text for t in buckets[0]] == ["A"]
    assert [t.text for t in buckets[1]] == ["B"]
