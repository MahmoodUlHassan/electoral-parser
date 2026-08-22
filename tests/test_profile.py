from __future__ import annotations

from parser.profile import PhaseProfiler


def test_phase_profiler_accumulates_and_ranks():
    p = PhaseProfiler()
    with p.track("OCR card", count=2):
        pass
    p.add("PDF render", 1.5, count=3)
    p.add("OCR card", 0.4, count=1)
    p.add_detail("OCR detect (sum)", 2.0, count=5)
    p.add_detail("OCR recognize (sum)", 0.5, count=5)
    d = p.to_dict()
    assert d["totalSeconds"] >= 1.5
    names = [row["name"] for row in d["phases"]]
    assert names[0] == "PDF render"
    assert p.counts["OCR card"] == 3
    assert d["ocrInternals"]["phases"][0]["name"] == "OCR detect (sum)"
