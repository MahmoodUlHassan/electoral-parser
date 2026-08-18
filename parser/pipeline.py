from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from detection.bands import detect_body_line_regions
from detection.classifier import classify_page, detect_cards
from detection.grid import detect_content_frame
from detection.visualize import crop_relative, draw_detections, save_card_crop
from exporters.writers import export_voters
from extractors.header import merge_meta, parse_cover_totals, parse_page_header
from extractors.voter import apply_missing_band_text, merge_card_fields, parse_labeled_lines, parse_voter_card
from ocr.engine import PaddleOcrEngine
from ocr.types import OcrEngine
from parser.config import DEFAULT_LAYOUT, LayoutProfile, NATIVE_EXTRACT
from parser.models import OcrToken, PageMeta, PageType, VoterRecord
from preprocessing.renderer import render_page, save_debug_image
from validators.rules import validate_voter

logger = logging.getLogger("electoral.pipeline")


@dataclass(slots=True)
class RunPaths:
    json_dir: Path
    csv_dir: Path
    debug_dir: Path
    logs_dir: Path

    def ensure(self) -> None:
        for p in (self.json_dir, self.csv_dir, self.debug_dir, self.logs_dir):
            p.mkdir(parents=True, exist_ok=True)


def default_run_paths(root: Path) -> RunPaths:
    return RunPaths(
        json_dir=root / "json",
        csv_dir=root / "csv",
        debug_dir=root / "debug",
        logs_dir=root / "logs",
    )


def parse_page_spec(spec: str | None, page_count: int) -> set[int] | None:
    if not spec:
        return None
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            pages.update(range(int(a), int(b) + 1))
        else:
            pages.add(int(part))
    return {p for p in pages if 1 <= p <= page_count}


def _upscale(image: np.ndarray, factor: float = 2.0) -> np.ndarray:
    if factor == 1.0:
        return image
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def _ocr_band(ocr: OcrEngine, image: np.ndarray, region) -> str:
    crop = crop_relative(image, region)
    if crop.size == 0:
        return ""
    try:
        return " ".join(ocr.recognize(crop).texts)
    except Exception:
        logger.exception("band OCR failed")
        return ""


def refill_missing_fields(
    record: VoterRecord,
    ocr: OcrEngine,
    ocr_image: np.ndarray,
    layout: LayoutProfile = DEFAULT_LAYOUT,
) -> VoterRecord:
    record = merge_card_fields(record, parse_labeled_lines(record.raw_ocr))
    missing = (
        record.house_no is None
        or record.age is None
        or record.gender is None
        or record.relation_type is None
        or record.relative_name is None
        or record.name is None
    )
    if not missing:
        return record
    regions = detect_body_line_regions(ocr_image, layout)
    if not regions:
        for region in (layout.house_region, layout.age_region, layout.age_fallback_region):
            if record.house_no is None:
                record = apply_missing_band_text(
                    record, house_text=_ocr_band(ocr, ocr_image, region)
                )
            if record.age is None or record.gender is None:
                record = apply_missing_band_text(
                    record, age_text=_ocr_band(ocr, ocr_image, region)
                )
        return record
    lines = [_ocr_band(ocr, ocr_image, region) for region in regions]
    return merge_card_fields(record, parse_labeled_lines(lines), extra_lines=lines)


def _record_from_export(row: dict) -> VoterRecord:
    tokens = [
        OcrToken(
            text=str(item.get("text") or ""),
            confidence=float(item.get("confidence") or 0),
            bbox=item.get("bbox") or [[0.0, 0.0]] * 4,
        )
        for item in (row.get("boundingBoxes") or [])
        if item.get("text")
    ]
    return VoterRecord(
        serial_no=row.get("serialNo"),
        epic=row.get("epic"),
        name=row.get("name"),
        relation_type=row.get("relationType"),
        relative_name=row.get("relativeName"),
        house_no=row.get("houseNo"),
        age=row.get("age"),
        gender=row.get("gender"),
        page=int(row.get("page") or 0),
        part_no=row.get("partNo"),
        constituency=row.get("constituency"),
        section=row.get("section"),
        raw_ocr=list(row.get("rawOcr") or []),
        tokens=tokens,
        confidence=float(row.get("confidence") or 0),
        crop_path=row.get("cropPath"),
        valid=bool(row.get("valid", True)),
        errors=list(row.get("errors") or []),
    )


def _resolve_crop(crop_path: str | None, root: Path) -> Path | None:
    if not crop_path:
        return None
    candidate = Path(crop_path)
    if candidate.is_file():
        return candidate
    nested = root / crop_path
    if nested.is_file():
        return nested
    return None


def refill_saved_json(
    json_path: Path,
    *,
    csv_path: Path | None = None,
    engine: OcrEngine | None = None,
    layout: LayoutProfile = DEFAULT_LAYOUT,
    project_root: Path | None = None,
) -> dict[str, int]:
    """Second-pass house/age OCR on existing card crops. Does not re-read the PDF."""
    json_path = json_path.expanduser().resolve()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = payload.get("voters") or []
    ocr = engine or PaddleOcrEngine(lang="en")
    root = project_root or Path.cwd()
    house_filled = age_filled = gender_filled = missing_crop = 0
    records: list[VoterRecord] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("refill bands", total=len(rows))
        for row in rows:
            rec = _record_from_export(row)
            before = (rec.house_no, rec.age, rec.gender, rec.relation_type, rec.relative_name)
            crop = _resolve_crop(rec.crop_path, root)
            if crop is None:
                missing_crop += 1
            else:
                image = cv2.imread(str(crop))
                if image is None:
                    missing_crop += 1
                else:
                    rec = refill_missing_fields(rec, ocr, _upscale(image, 2.0), layout)
            rec = validate_voter(rec, layout)
            if before[0] is None and rec.house_no:
                house_filled += 1
            if before[1] is None and rec.age is not None:
                age_filled += 1
            if before[2] is None and rec.gender:
                gender_filled += 1
            records.append(rec)
            progress.advance(task)
    extra = {key: value for key, value in payload.items() if key != "voters"}
    extra["validCount"] = sum(1 for rec in records if rec.valid)
    extra["extractedTotal"] = len(records)
    if csv_path is None:
        csv_path = json_path.parent.parent / "csv" / "voters.csv"
    export_voters(records, json_path, csv_path, extra=extra)
    logger.info(
        "refill %s house=%s age=%s gender=%s missing_crop=%s valid=%s/%s",
        json_path,
        house_filled,
        age_filled,
        gender_filled,
        missing_crop,
        extra["validCount"],
        len(records),
    )
    return {
        "total": len(records),
        "houseFilled": house_filled,
        "ageFilled": age_filled,
        "genderFilled": gender_filled,
        "missingCrop": missing_crop,
        "valid": int(extra["validCount"]),
    }


def run_pipeline(
    pdf_path: Path,
    paths: RunPaths,
    *,
    visualize: bool = True,
    pages: str | None = None,
    dpi: int = 300,
    scale: float | None = None,
    prefer_native: bool = NATIVE_EXTRACT,
    engine: OcrEngine | None = None,
    layout: LayoutProfile = DEFAULT_LAYOUT,
) -> list[VoterRecord]:
    paths.ensure()
    pdf_path = pdf_path.expanduser().resolve()
    logger.info("Opening %s", pdf_path)
    doc = pymupdf.open(pdf_path)
    wanted = parse_page_spec(pages, doc.page_count)
    ocr = engine or PaddleOcrEngine(lang="en")
    voters: list[VoterRecord] = []
    global_meta = PageMeta()
    page_types: dict[int, str] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("pages", total=doc.page_count)
        for page_index in range(doc.page_count):
            page_no = page_index + 1
            if wanted is not None and page_no not in wanted:
                progress.advance(task)
                continue
            image = render_page(
                doc, page_index, dpi=dpi, scale=scale, prefer_native=prefer_native
            )
            page_debug = paths.debug_dir / "pages" / f"page{page_no}.png"
            save_debug_image(image, page_debug)

            page_type = classify_page(image, layout)
            page_types[page_no] = page_type.value
            logger.info("page %s → %s", page_no, page_type.value)

            if page_type == PageType.HEADER:
                cover = cv2.resize(image, (0, 0), fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
                cover_tokens = ocr.recognize(cover).tokens
                global_meta = merge_meta(global_meta, parse_cover_totals(cover_tokens))

            if page_type != PageType.VOTER:
                progress.advance(task)
                continue

            cards, method = detect_cards(image, layout)
            occupied = [c for c in cards if c.occupied]
            logger.info("page %s cards=%s method=%s", page_no, len(occupied), method.value)
            if visualize:
                vis = draw_detections(image, cards)
                save_debug_image(vis, paths.debug_dir / f"page{page_no}_detected.png")
            if len(occupied) < layout.min_full_page_cards:
                save_debug_image(image, paths.debug_dir / f"page{page_no}_low_count.png")

            h, w = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            frame = detect_content_frame(gray, layout)
            header_bottom = max(24, min(frame.y - 2, occupied[0].box.y - 2) if occupied else frame.y - 2)
            footer_top = min(h - 16, frame.y2 + 2)
            header_img = image[0:header_bottom, :]
            footer_img = image[footer_top:h, :]
            header_tokens = ocr.recognize(_upscale(header_img, 2.0)).tokens
            footer_tokens = ocr.recognize(_upscale(footer_img, 2.0)).tokens
            page_meta = parse_page_header(header_tokens + footer_tokens, page_no)
            global_meta = merge_meta(global_meta, page_meta)
            page_meta.constituency = page_meta.constituency or global_meta.constituency
            page_meta.part_no = page_meta.part_no or global_meta.part_no
            page_meta.section = page_meta.section or global_meta.section

            for card in occupied:
                crop_name = f"page{page_no}_card{card.index + 1}.png"
                crop_path = paths.debug_dir / "cards" / crop_name
                crop = save_card_crop(image, card, crop_path)
                ocr_image = _upscale(crop, 2.0)
                result = ocr.recognize(ocr_image)
                record = parse_voter_card(
                    result.tokens,
                    page=page_no,
                    meta=page_meta,
                    crop_width=ocr_image.shape[1],
                    crop_height=ocr_image.shape[0],
                    crop_path=str(crop_path),
                    mean_confidence=result.mean_confidence,
                )
                record = refill_missing_fields(record, ocr, ocr_image, layout)
                record = validate_voter(record, layout)
                if record.epic is None and record.name is None:
                    logger.debug("skip empty cell page=%s card=%s", page_no, card.index + 1)
                    continue
                voters.append(record)
            progress.advance(task)

    voters.sort(key=lambda v: (v.serial_no is None, v.serial_no or 0, v.page))
    extra = {
        "source": str(pdf_path),
        "layout": layout.name,
        "pageTypes": page_types,
        "partNo": global_meta.part_no,
        "constituency": global_meta.constituency,
        "expectedTotal": global_meta.expected_total,
        "extractedTotal": len(voters),
        "validCount": sum(1 for v in voters if v.valid),
    }
    export_voters(
        voters,
        paths.json_dir / "voters.json",
        paths.csv_dir / "voters.csv",
        extra=extra,
    )
    logger.info(
        "extracted %s voters (%s valid) expected=%s",
        len(voters),
        extra["validCount"],
        global_meta.expected_total,
    )
    return voters
