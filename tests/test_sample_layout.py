from __future__ import annotations

import os

import pymupdf

from detection.classifier import classify_page, detect_cards
from preprocessing.renderer import render_page


def test_sample_pdf_page_types():
    path = "/Users/mahmoodulhassan/Downloads/2026-EROLLGEN-S29-52-SIR-DraftRoll-Revision1-ENG-419-WI.pdf"
    if not os.path.exists(path):
        import pytest

        pytest.skip("sample electoral roll PDF not present")
    doc = pymupdf.open(path)
    expected = {
        1: "header",
        2: "map",
        3: "voter",
        22: "voter",
        30: "voter",
        31: "supplement",
    }
    for page_no, want in expected.items():
        img = render_page(doc, page_no - 1)
        got = classify_page(img).value
        assert got == want, f"page {page_no}: {got} != {want}"


def test_sample_pdf_grid_page3():
    path = "/Users/mahmoodulhassan/Downloads/2026-EROLLGEN-S29-52-SIR-DraftRoll-Revision1-ENG-419-WI.pdf"
    if not os.path.exists(path):
        import pytest

        pytest.skip("sample electoral roll PDF not present")
    doc = pymupdf.open(path)
    img = render_page(doc, 2)
    cards, method = detect_cards(img)
    assert method.value == "grid"
    assert 28 <= len(cards) <= 30
