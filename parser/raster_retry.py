"""Retry invalid voter pages from debug page rasters (no PDF)."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from detection.classifier import detect_cards
from detection.visualize import (
    crop_card,
    crop_card_for_ocr,
    crop_card_padded,
    save_card_crop,
)
from exporters.writers import export_voters
from extractors.voter import merge_card_fields, parse_labeled_lines, parse_voter_card
from ocr.engine import DEFAULT_OCR_MODEL, PaddleOcrEngine
from ocr.types import OcrEngine
from parser.config import DEFAULT_LAYOUT, LayoutProfile
from parser.models import PageMeta, VoterRecord
from parser.pipeline import (
    RunPaths,
    _process_voter_page_row,
    _record_from_export,
    _upscale,
    needs_band_refill,
    refill_age_from_expanded_crop,
    refill_missing_fields,
    retry_invalid_with_small,
)
from validators.rules import validate_voter

logger = logging.getLogger("electoral.raster_retry")


def is_age_gender_clip_only(rec: VoterRecord) -> bool:
    """True when failures are exactly missing age+gender (6-line wrap / Age clipped)."""
    if rec.valid:
        return False
    return set(rec.errors or []) == {"missing_age", "missing_gender"}


def _meta_from_voters(voters: list[VoterRecord], page_no: int) -> PageMeta:
    sample = next((v for v in voters if v.page == page_no), None) or (
        voters[0] if voters else None
    )
    if sample is None:
        return PageMeta(page=page_no)
    return PageMeta(
        page=page_no,
        part_no=sample.part_no,
        constituency=sample.constituency,
        section=sample.section,
    )


def _labeled_fields(raw_ocr) -> dict:
    if raw_ocr is None:
        return {}
    if isinstance(raw_ocr, str):
        lines = [ln.strip() for ln in raw_ocr.splitlines() if ln.strip()]
    else:
        lines = [str(x).strip() for x in raw_ocr if str(x).strip()]
    return parse_labeled_lines(lines)


def _ocr_voter_page_card(
    image: np.ndarray,
    page_no: int,
    page_meta: PageMeta,
    occupied: list,
    ocr: OcrEngine,
    layout: LayoutProfile,
    paths: RunPaths,
    *,
    ocr_workers: int,
) -> tuple[list[VoterRecord], int, int]:
    refill_calls = 0
    refill_skips = 0
    sparse_page = len(occupied) < layout.min_full_page_cards
    prepared: list[tuple] = []
    for card in occupied:
        crop_name = f"page{page_no}_card{card.index + 1}.png"
        crop_path = paths.debug_dir / "cards" / crop_name
        save_card_crop(image, card, crop_path)
        if sparse_page:
            ocr_src = crop_card_padded(image, card)
        else:
            ocr_src = crop_card_for_ocr(image, card, layout)
        prepared.append((card, crop_path, _upscale(ocr_src, 2.0)))

    if ocr_workers == 1 or len(prepared) <= 1:
        ocr_results = [ocr.recognize(img) for _, _, img in prepared]
    else:
        with ThreadPoolExecutor(max_workers=ocr_workers) as pool:
            ocr_results = list(pool.map(lambda item: ocr.recognize(item[2]), prepared))

    out: list[VoterRecord] = []
    for (card, crop_path, ocr_image), result in zip(prepared, ocr_results):
        record = parse_voter_card(
            result.tokens,
            page=page_no,
            meta=page_meta,
            crop_width=ocr_image.shape[1],
            crop_height=ocr_image.shape[0],
            crop_path=str(crop_path),
            mean_confidence=result.mean_confidence,
        )
        record = merge_card_fields(record, _labeled_fields(record.raw_ocr))
        if needs_band_refill(record):
            if sparse_page:
                full_2x = _upscale(crop_card_padded(image, card), 2.0)
            else:
                full_2x = _upscale(crop_card(image, card), 2.0)
            record = refill_missing_fields(record, ocr, full_2x, layout)
            if record.age is None or record.gender is None:
                record = refill_age_from_expanded_crop(
                    record, ocr, image, card.box, layout
                )
            refill_calls += 1
        else:
            refill_skips += 1
        record = validate_voter(record, layout)
        if record.epic is None and record.name is None:
            continue
        out.append(record)
    return out, refill_calls, refill_skips


def retry_invalid_from_rasters(
    paths: RunPaths,
    *,
    json_path: Path | None = None,
    layout: LayoutProfile = DEFAULT_LAYOUT,
    stats: dict | None = None,
    ocr_workers: int = 1,
    ocr_model: str = DEFAULT_OCR_MODEL,
    retry_small: bool = True,
    ocr_mode: str = "card",
    skip_age_gender_clip: bool = True,
) -> list[VoterRecord]:
    """Re-OCR selected invalid pages from debug/pages/pageN.png (no PDF).

    Pages whose only invalids are missing age/gender are skipped when
    skip_age_gender_clip is True.
    """
    paths.ensure()
    json_path = json_path or (paths.json_dir / "voters.json")
    if not json_path.is_file():
        raise FileNotFoundError(f"No existing parse at {json_path}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    existing = [_record_from_export(row) for row in (payload.get("voters") or [])]
    invalids = [v for v in existing if not v.valid]
    if skip_age_gender_clip:
        retry_invalids = [v for v in invalids if not is_age_gender_clip_only(v)]
        skipped_clip = len(invalids) - len(retry_invalids)
    else:
        retry_invalids = list(invalids)
        skipped_clip = 0

    retry_pages = sorted({int(v.page) for v in retry_invalids if v.page})
    if not retry_pages:
        logger.info(
            "raster-retry %s: nothing to do (invalid=%s clip_skipped=%s)",
            json_path,
            len(invalids),
            skipped_clip,
        )
        if stats is not None:
            stats.update(
                {
                    "extractedTotal": len(existing),
                    "validCount": sum(1 for v in existing if v.valid),
                    "expectedTotal": payload.get("expectedTotal"),
                    "retriedPages": [],
                    "clipSkippedInvalids": skipped_clip,
                }
            )
        return existing

    ocr_model = (ocr_model or DEFAULT_OCR_MODEL).lower()
    ocr_mode = (ocr_mode or "card").lower()
    ocr = PaddleOcrEngine(lang="en", model=ocr_model)
    ocr._ensure()
    small: OcrEngine | None = None
    if ocr_mode == "row" and retry_small and ocr_model != "small":
        small = PaddleOcrEngine(lang="en", model="small")
        small._ensure()

    before_valid = sum(1 for v in existing if v.valid)
    refreshed_by_page: dict[int, list[VoterRecord]] = {}
    missing_rasters: list[int] = []

    for page_no in retry_pages:
        raster = paths.debug_dir / "pages" / f"page{page_no}.png"
        if not raster.is_file():
            logger.warning("missing raster %s — skip page %s", raster, page_no)
            missing_rasters.append(page_no)
            continue
        image = cv2.imread(str(raster))
        if image is None:
            logger.warning("failed to read %s", raster)
            missing_rasters.append(page_no)
            continue

        cards, method = detect_cards(image, layout)
        occupied = [c for c in cards if c.occupied]
        logger.info(
            "raster page %s cards=%s method=%s", page_no, len(occupied), method.value
        )
        page_meta = _meta_from_voters(existing, page_no)
        if ocr_mode == "row":
            page_voters, _, _ = _process_voter_page_row(
                image,
                page_no,
                page_meta,
                occupied,
                ocr,
                layout,
                paths,
                small=small,
            )
        else:
            page_voters, _, _ = _ocr_voter_page_card(
                image,
                page_no,
                page_meta,
                occupied,
                ocr,
                layout,
                paths,
                ocr_workers=ocr_workers,
            )
        refreshed_by_page[page_no] = page_voters
        logger.info(
            "raster-retry page %s: %s voters (%s valid)",
            page_no,
            len(page_voters),
            sum(1 for v in page_voters if v.valid),
        )

    kept = [v for v in existing if v.page not in refreshed_by_page]
    merged = kept + [
        v for page in sorted(refreshed_by_page) for v in refreshed_by_page[page]
    ]
    merged.sort(key=lambda v: (v.serial_no is None, v.serial_no or 0, v.page))

    small_attempted = small_fixed = 0
    if retry_small and ocr_model != "small" and any(
        (not v.valid) and not is_age_gender_clip_only(v) for v in merged
    ):
        merged, small_attempted, small_fixed = retry_invalid_with_small(
            merged,
            layout=layout,
            project_root=Path.cwd(),
        )

    after_valid = sum(1 for v in merged if v.valid)
    extra = {
        k: payload.get(k)
        for k in (
            "source",
            "layout",
            "pageTypes",
            "partNo",
            "constituency",
            "expectedTotal",
            "ocrModel",
            "ocrMode",
            "ocrWorkers",
        )
        if k in payload
    }
    extra.update(
        {
            "extractedTotal": len(merged),
            "validCount": after_valid,
            "retriedPages": sorted(refreshed_by_page),
            "clipSkippedInvalids": skipped_clip,
            "missingRasters": missing_rasters,
            "ocrMode": f"raster-{ocr_mode}",
            "smallRetryAttempted": small_attempted,
            "smallRetryFixed": small_fixed,
        }
    )
    export_voters(merged, json_path, paths.csv_dir / "voters.csv", extra=extra)
    if stats is not None:
        stats.update(extra)
        stats["validBefore"] = before_valid
        stats["validAfter"] = after_valid
        stats["validDelta"] = after_valid - before_valid
    logger.info(
        "raster-retry done valid %s → %s (clip_skipped=%s pages=%s missing=%s)",
        before_valid,
        after_valid,
        skipped_clip,
        sorted(refreshed_by_page),
        missing_rasters,
    )
    return merged
