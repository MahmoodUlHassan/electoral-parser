from __future__ import annotations

from ocr.row_strip import assign_tokens_by_x, group_cards_into_rows, row_union_box
from parser.models import Box, CardDetection, DetectionMethod, OcrToken


def test_group_cards_into_rows():
    cards = [
        CardDetection(i, Box((i % 3) * 100, (i // 3) * 50, 90, 40), True, DetectionMethod.GRID)
        for i in range(6)
    ]
    rows = group_cards_into_rows(cards)
    assert len(rows) == 2
    assert [c.index for c in rows[0]] == [0, 1, 2]
    assert [c.index for c in rows[1]] == [3, 4, 5]


def test_assign_tokens_by_x():
    row = [
        CardDetection(0, Box(0, 10, 100, 40), True, DetectionMethod.GRID),
        CardDetection(1, Box(110, 10, 100, 40), True, DetectionMethod.GRID),
        CardDetection(2, Box(220, 10, 100, 40), True, DetectionMethod.GRID),
    ]
    union = row_union_box(row)
    tokens = [
        OcrToken("L", 0.9, [[20, 5], [40, 5], [40, 15], [20, 15]]),
        OcrToken("M", 0.9, [[150, 5], [170, 5], [170, 15], [150, 15]]),
        OcrToken("R", 0.9, [[250, 5], [270, 5], [270, 15], [250, 15]]),
    ]
    buckets = assign_tokens_by_x(tokens, row, union)
    assert [t.text for t in buckets[0]] == ["L"]
    assert [t.text for t in buckets[1]] == ["M"]
    assert [t.text for t in buckets[2]] == ["R"]
