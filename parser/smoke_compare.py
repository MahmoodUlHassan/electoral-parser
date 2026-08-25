"""Side-by-side smoke: per-card OCR vs page OCR → assign (experiment)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from detection.classifier import detect_cards
from detection.visualize import crop_card, crop_card_for_ocr, crop_card_padded
from extractors.voter import merge_card_fields, parse_labeled_lines, parse_voter_card
from ocr.assign import (
    adaptive_ocr_scale,
    assign_tokens_to_cards,
    scale_token_to_page,
    tokens_to_card_local,
)
from ocr.engine import PaddleOcrEngine
from ocr.row_strip import (
    assign_tokens_by_x,
    group_cards_into_rows,
    prepare_row_strip,
    row_ocr_scale,
    tokens_to_card_local_from_strip,
)
from parser.config import DEFAULT_LAYOUT, LayoutProfile, NATIVE_EXTRACT
from parser.models import PageMeta, VoterRecord
from parser.pipeline import needs_band_refill, refill_missing_fields
from preprocessing.renderer import render_page
from validators.rules import validate_voter

logger = logging.getLogger("electoral.smoke")


@dataclass(slots=True)
class PathStats:
    name: str
    seconds: float
    ocr_calls: int
    scale: float
    extracted: int
    valid: int
    missing_epic: int
    missing_age: int
    missing_house: int
    voters: list[VoterRecord]
    page: int = 0


def _upscale(image: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return image
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def _summarize(
    name: str,
    seconds: float,
    ocr_calls: int,
    scale: float,
    voters: list[VoterRecord],
    page: int = 0,
) -> PathStats:
    return PathStats(
        name=name,
        seconds=seconds,
        ocr_calls=ocr_calls,
        scale=scale,
        extracted=len(voters),
        valid=sum(1 for v in voters if v.valid),
        missing_epic=sum(1 for v in voters if v.epic is None),
        missing_age=sum(1 for v in voters if v.age is None),
        missing_house=sum(1 for v in voters if v.house_no is None),
        voters=voters,
        page=page,
    )


def run_card_path(
    image: np.ndarray,
    page_no: int,
    ocr: PaddleOcrEngine,
    layout: LayoutProfile = DEFAULT_LAYOUT,
    meta: PageMeta | None = None,
) -> PathStats:
    """Current production path: one OCR call per occupied card @ fixed 2×."""
    meta = meta or PageMeta()
    t0 = time.perf_counter()
    cards, _ = detect_cards(image, layout)
    occupied = [c for c in cards if c.occupied]
    scale = 2.0
    ocr_calls = 0
    voters: list[VoterRecord] = []
    for card in occupied:
        ocr_image = _upscale(crop_card_for_ocr(image, card, layout), scale)
        result = ocr.recognize(ocr_image)
        ocr_calls += 1
        rec = parse_voter_card(
            result.tokens,
            page=page_no,
            meta=meta,
            crop_width=ocr_image.shape[1],
            crop_height=ocr_image.shape[0],
            mean_confidence=result.mean_confidence,
        )
        rec = merge_card_fields(rec, parse_labeled_lines(rec.raw_ocr))
        rec = validate_voter(rec, layout)
        if rec.epic is None and rec.name is None:
            continue
        voters.append(rec)
    return _summarize("card×30@2x", time.perf_counter() - t0, ocr_calls, scale, voters, page_no)


def run_page_assign_path(
    image: np.ndarray,
    page_no: int,
    ocr: PaddleOcrEngine,
    layout: LayoutProfile = DEFAULT_LAYOUT,
    meta: PageMeta | None = None,
) -> PathStats:
    """Page OCR once → assign tokens by bbox overlap → parse per card."""
    meta = meta or PageMeta()
    t0 = time.perf_counter()
    cards, _ = detect_cards(image, layout)
    occupied = [c for c in cards if c.occupied]
    if not occupied:
        return _summarize("page→assign", time.perf_counter() - t0, 0, 1.0, [], page_no)

    median_h = float(np.median([c.box.h for c in occupied]))
    scale = adaptive_ocr_scale(median_h)
    page_ocr = _upscale(image, scale)
    h, w = page_ocr.shape[:2]
    # Paddle max_side ~4000; log if we still blow it.
    max_side = max(h, w)
    if max_side > 4000:
        logger.warning(
            "page OCR side %s > 4000 — Paddle will downscale (card_h=%.0f scale=%s)",
            max_side,
            median_h,
            scale,
        )

    result = ocr.recognize(page_ocr)
    ocr_calls = 1
    page_tokens = [scale_token_to_page(t, scale) for t in result.tokens]
    buckets = assign_tokens_to_cards(page_tokens, occupied, layout)

    voters: list[VoterRecord] = []
    for card in occupied:
        tokens = tokens_to_card_local(buckets.get(card.index, []), card)
        # parse_voter_card → group_lines sorts by y then rebuilds lines from bboxes
        rec = parse_voter_card(
            tokens,
            page=page_no,
            meta=meta,
            crop_width=card.box.w,
            crop_height=card.box.h,
            mean_confidence=result.mean_confidence,
        )
        rec = merge_card_fields(rec, parse_labeled_lines(rec.raw_ocr))
        rec = validate_voter(rec, layout)
        if rec.epic is None and rec.name is None:
            continue
        voters.append(rec)

    label = f"page→assign@{scale:g}x"
    return _summarize(label, time.perf_counter() - t0, ocr_calls, scale, voters, page_no)


def _parse_card_tokens(
    tokens: list,
    *,
    page_no: int,
    meta: PageMeta,
    card,
    mean_conf: float,
    layout: LayoutProfile,
) -> VoterRecord:
    rec = parse_voter_card(
        tokens,
        page=page_no,
        meta=meta,
        crop_width=card.box.w,
        crop_height=card.box.h,
        mean_confidence=mean_conf,
    )
    rec = merge_card_fields(rec, parse_labeled_lines(rec.raw_ocr))
    return validate_voter(rec, layout)


def _recover_invalid_card(
    record: VoterRecord,
    image: np.ndarray,
    card,
    page_no: int,
    meta: PageMeta,
    ocr: PaddleOcrEngine,
    layout: LayoutProfile,
    *,
    small: PaddleOcrEngine | None,
) -> tuple[VoterRecord, int]:
    """Failure ladder: card re-OCR @2× → band refill → optional small."""
    calls = 0
    full_2x = _upscale(crop_card_for_ocr(image, card, layout), 2.0)
    result = ocr.recognize(full_2x)
    calls += 1
    rec = parse_voter_card(
        result.tokens,
        page=page_no,
        meta=meta,
        crop_width=full_2x.shape[1],
        crop_height=full_2x.shape[0],
        mean_confidence=result.mean_confidence,
    )
    rec = merge_card_fields(rec, parse_labeled_lines(rec.raw_ocr))
    if needs_band_refill(rec):
        rec = refill_missing_fields(rec, ocr, _upscale(crop_card(image, card), 2.0), layout)
    rec = validate_voter(rec, layout)
    if rec.valid or small is None:
        return rec, calls

    small_img = _upscale(crop_card_padded(image, card), 2.0)
    result = small.recognize(small_img)
    calls += 1
    updated = parse_voter_card(
        result.tokens,
        page=page_no,
        meta=meta,
        crop_width=small_img.shape[1],
        crop_height=small_img.shape[0],
        mean_confidence=result.mean_confidence,
    )
    updated = merge_card_fields(updated, parse_labeled_lines(updated.raw_ocr))
    if needs_band_refill(updated):
        updated = refill_missing_fields(
            updated, small, _upscale(crop_card(image, card), 2.0), layout
        )
    updated = validate_voter(updated, layout)
    return updated, calls


def run_row_path(
    image: np.ndarray,
    page_no: int,
    ocr: PaddleOcrEngine,
    layout: LayoutProfile = DEFAULT_LAYOUT,
    meta: PageMeta | None = None,
    *,
    recover: bool = True,
    small: PaddleOcrEngine | None = None,
) -> PathStats:
    """10 row strips → wipe photo/borders → OCR → assign by x → parse (+ recover)."""
    meta = meta or PageMeta()
    t0 = time.perf_counter()
    cards, _ = detect_cards(image, layout)
    occupied = [c for c in cards if c.occupied]
    rows = group_cards_into_rows(cards, layout)
    ocr_calls = 0
    scale_used = 1.0
    by_index: dict[int, VoterRecord] = {}

    for row_cards in rows:
        strip, union = prepare_row_strip(image, row_cards, layout)
        scale = row_ocr_scale(row_cards)
        scale_used = scale
        ocr_img = _upscale(strip, scale)
        result = ocr.recognize(ocr_img)
        ocr_calls += 1
        # Map tokens back to strip-native coords
        tokens = [scale_token_to_page(t, scale) for t in result.tokens]
        buckets = assign_tokens_by_x(tokens, row_cards, union)
        for card in row_cards:
            local = tokens_to_card_local_from_strip(buckets.get(card.index, []), card, union)
            rec = _parse_card_tokens(
                local,
                page_no=page_no,
                meta=meta,
                card=card,
                mean_conf=result.mean_confidence,
                layout=layout,
            )
            if rec.epic is None and rec.name is None:
                continue
            if recover and not rec.valid:
                rec, extra = _recover_invalid_card(
                    rec, image, card, page_no, meta, ocr, layout, small=small
                )
                ocr_calls += extra
            by_index[card.index] = rec

    voters = [by_index[i] for i in sorted(by_index)]
    label = f"row×10@{scale_used:g}x"
    if recover:
        label += "+recover"
    return _summarize(label, time.perf_counter() - t0, ocr_calls, scale_used, voters, page_no)


def compare_page(
    pdf_path: Path,
    page_no: int,
    *,
    ocr_model: str = "tiny",
    prefer_native: bool = NATIVE_EXTRACT,
    engine: PaddleOcrEngine | None = None,
    recover: bool = True,
) -> tuple[PathStats, PathStats, PathStats]:
    pdf_path = pdf_path.expanduser().resolve()
    doc = pymupdf.open(pdf_path)
    if page_no < 1 or page_no > doc.page_count:
        raise ValueError(f"page {page_no} out of range 1..{doc.page_count}")
    image = render_page(doc, page_no - 1, prefer_native=prefer_native)
    doc.close()

    ocr = engine or PaddleOcrEngine(lang="en", model=ocr_model)
    if engine is None:
        ocr._ensure()
    small = None
    if recover:
        small = PaddleOcrEngine(lang="en", model="small")
        small._ensure()
    meta = PageMeta(page=page_no)

    card_stats = run_card_path(image, page_no, ocr, meta=meta)
    page_stats = run_page_assign_path(image, page_no, ocr, meta=meta)
    row_stats = run_row_path(
        image, page_no, ocr, meta=meta, recover=recover, small=small
    )
    return card_stats, page_stats, row_stats


def scan_pdf_page_assign(
    pdf_path: Path,
    *,
    ocr_model: str = "tiny",
    prefer_native: bool = NATIVE_EXTRACT,
    pages: set[int] | None = None,
) -> list[PathStats]:
    """Run page→assign on every (or selected) page; skip empty non-voter results lightly."""
    from detection.classifier import classify_page
    from parser.models import PageType

    pdf_path = pdf_path.expanduser().resolve()
    doc = pymupdf.open(pdf_path)
    ocr = PaddleOcrEngine(lang="en", model=ocr_model)
    ocr._ensure()
    results: list[PathStats] = []
    t_all = time.perf_counter()
    for page_index in range(doc.page_count):
        page_no = page_index + 1
        if pages is not None and page_no not in pages:
            continue
        image = render_page(doc, page_index, prefer_native=prefer_native)
        page_type = classify_page(image, DEFAULT_LAYOUT)
        if page_type != PageType.VOTER:
            logger.info("page %s → %s (skip OCR)", page_no, page_type.value)
            continue
        stats = run_page_assign_path(image, page_no, ocr, meta=PageMeta(page=page_no))
        results.append(stats)
        logger.info(
            "page %s %s",
            page_no,
            format_stats(stats),
        )
    doc.close()
    wall = time.perf_counter() - t_all
    total_v = sum(s.valid for s in results)
    total_e = sum(s.extracted for s in results)
    logger.info(
        "PDF page→assign done pages=%s wall=%.1fs voters valid=%s/%s",
        len(results),
        wall,
        total_v,
        total_e,
    )
    return results


def format_stats(s: PathStats) -> str:
    return (
        f"{s.name:22s}  {s.seconds:6.2f}s  ocr_calls={s.ocr_calls}  scale={s.scale:g}  "
        f"valid={s.valid}/{s.extracted}  "
        f"miss_epic={s.missing_epic} miss_age={s.missing_age} miss_house={s.missing_house}"
    )
