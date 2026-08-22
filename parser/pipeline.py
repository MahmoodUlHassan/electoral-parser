from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from detection.bands import detect_body_line_regions
from detection.classifier import classify_page, detect_cards
from detection.grid import detect_content_frame
from detection.visualize import (
    crop_card,
    crop_card_for_ocr,
    crop_card_padded,
    crop_relative,
    draw_detections,
    save_card_crop,
)
from exporters.writers import export_voters
from extractors.header import merge_meta, parse_cover_totals, parse_page_header
from extractors.voter import (
    apply_missing_band_text,
    merge_card_fields,
    parse_labeled_lines,
    parse_voter_card,
)
from ocr.engine import DEFAULT_OCR_MODEL, OcrEnginePool, PaddleOcrEngine
from ocr.types import OcrEngine, OcrResult
from parser.config import DEFAULT_LAYOUT, LayoutProfile, NATIVE_EXTRACT, RelativeBox
from parser.models import OcrToken, PageMeta, PageType, VoterRecord
from parser.profile import PhaseProfiler
from preprocessing.renderer import render_page, save_debug_image
from validators.rules import validate_voter
from rich.console import Console

logger = logging.getLogger("electoral.pipeline")
_profile_console = Console(stderr=True)


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


def _ocr_band(
    ocr: OcrEngine,
    image: np.ndarray,
    region,
    profiler: PhaseProfiler | None = None,
    *,
    scale: float = 1.0,
    phase: str = "OCR band refill",
) -> str:
    crop = crop_relative(image, region)
    if crop.size == 0:
        return ""
    if scale != 1.0:
        crop = _upscale(crop, scale)
    try:
        if profiler is not None:
            with profiler.track(phase):
                return " ".join(ocr.recognize(crop).texts)
        return " ".join(ocr.recognize(crop).texts)
    except Exception:
        logger.exception("band OCR failed")
        return ""


def needs_body_refill(record: VoterRecord) -> bool:
    return (
        record.house_no is None
        or record.age is None
        or record.gender is None
        or record.relation_type is None
        or record.relative_name is None
        or record.name is None
    )


def needs_band_refill(record: VoterRecord) -> bool:
    """True when first-pass OCR left gaps that targeted re-OCR can fill."""
    return record.epic is None or needs_body_refill(record)


def _force_age_band_ocr(
    record: VoterRecord,
    ocr: OcrEngine,
    ocr_image: np.ndarray,
    layout: LayoutProfile,
    profiler: PhaseProfiler | None = None,
) -> VoterRecord:
    """OCR fixed Age strips when body bands missed Age (esp. 6-line wraps)."""
    if record.age is not None and record.gender is not None:
        return record
    for region in (
        layout.age_region,
        layout.age_fallback_region,
        layout.age_crowded_region,
    ):
        record = apply_missing_band_text(
            record,
            age_text=_ocr_band(ocr, ocr_image, region, profiler, phase="OCR age force"),
        )
        if record.age is not None and record.gender is not None:
            break
    return record


def refill_missing_fields(
    record: VoterRecord,
    ocr: OcrEngine,
    ocr_image: np.ndarray,
    layout: LayoutProfile = DEFAULT_LAYOUT,
    profiler: PhaseProfiler | None = None,
) -> VoterRecord:
    record = merge_card_fields(record, parse_labeled_lines(record.raw_ocr))
    if record.epic is None:
        # Full-card canvas (caller passes 2×); extra 1.5× → ~3× native on EPIC strips.
        for region, phase in (
            (layout.epic_region, "OCR epic strip"),
            (layout.epic_fallback_region, "OCR epic fallback"),
        ):
            epic_text = _ocr_band(
                ocr,
                ocr_image,
                region,
                profiler,
                scale=1.5,
                phase=phase,
            )
            record = apply_missing_band_text(record, epic_text=epic_text)
            if record.epic is not None:
                break
    if not needs_body_refill(record):
        return record
    with _maybe_track(profiler, "Band line detect"):
        regions = detect_body_line_regions(ocr_image, layout)
    if not regions:
        for region in (layout.house_region, layout.age_region, layout.age_fallback_region):
            if record.house_no is None:
                record = apply_missing_band_text(
                    record, house_text=_ocr_band(ocr, ocr_image, region, profiler)
                )
            if record.age is None or record.gender is None:
                record = apply_missing_band_text(
                    record, age_text=_ocr_band(ocr, ocr_image, region, profiler)
                )
        return _force_age_band_ocr(record, ocr, ocr_image, layout, profiler)

    lines = [_ocr_band(ocr, ocr_image, region, profiler) for region in regions]
    with _maybe_track(profiler, "Parse after refill"):
        record = merge_card_fields(record, parse_labeled_lines(lines), extra_lines=lines)
    # 6-line wraps fill all ink bands with Name/Relation/House — Age never appears
    # as its own band. Always force fixed Age strips when still missing.
    if record.age is None or record.gender is None or len(regions) >= 6:
        record = _force_age_band_ocr(record, ocr, ocr_image, layout, profiler)
    return record


def refill_age_from_expanded_crop(
    record: VoterRecord,
    ocr: OcrEngine,
    page_image: np.ndarray,
    card_box,
    layout: LayoutProfile = DEFAULT_LAYOUT,
    profiler: PhaseProfiler | None = None,
    *,
    expand_frac: float = 0.18,
) -> VoterRecord:
    """Re-crop slightly below the grid cell when Age was clipped by row pitch."""
    if record.age is not None and record.gender is not None:
        return record
    h, w = page_image.shape[:2]
    extra = max(8, int(card_box.h * expand_frac))
    y0 = max(0, card_box.y)
    y1 = min(h, card_box.y2 + extra)
    x0 = max(0, card_box.x)
    x1 = min(w, card_box.x2)
    if y1 <= card_box.y2 + 2:
        return record
    expanded = page_image[y0:y1, x0:x1]
    if expanded.size == 0:
        return record
    ocr_image = _upscale(expanded, 2.0)
    # Prefer the strip that was below the original cell.
    below_frac = card_box.h / max(expanded.shape[0], 1)
    below = RelativeBox(
        0.01,
        max(0.55, below_frac - 0.05),
        0.74,
        min(0.4, 1.0 - below_frac + 0.08),
    )
    record = apply_missing_band_text(
        record,
        age_text=_ocr_band(ocr, ocr_image, below, profiler, phase="OCR age expand"),
    )
    if record.age is None or record.gender is None:
        record = _force_age_band_ocr(record, ocr, ocr_image, layout, profiler)
    return record


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


def _resolve_retry_crop(crop_path: str | None, project_root: Path | None = None) -> Path | None:
    if not crop_path:
        return None
    candidate = Path(crop_path)
    if candidate.is_file():
        return candidate
    root = project_root or Path.cwd()
    nested = root / crop_path
    return nested if nested.is_file() else None


def _page_raster_for_crop(crop_path: Path, page: int, project_root: Path | None = None) -> Path | None:
    """debug/<stem>/cards/page37_card7.png → debug/<stem>/pages/page37.png"""
    pages_dir = crop_path.parent.parent / "pages"
    candidate = pages_dir / f"page{page}.png"
    if candidate.is_file():
        return candidate
    root = project_root or Path.cwd()
    nested = root / candidate
    return nested if nested.is_file() else None


def _retry_ocr_image(
    crop_path: Path,
    rec: VoterRecord,
    layout: LayoutProfile,
    project_root: Path | None = None,
) -> np.ndarray:
    """Prefer padded page re-crop when page raster was kept for invalids."""
    page_path = _page_raster_for_crop(crop_path, rec.page, project_root)
    if page_path is not None:
        page = cv2.imread(str(page_path))
        if page is not None:
            # Recover card index from filename page37_card7.png → index 6
            stem = crop_path.stem
            try:
                card_no = int(stem.split("_card")[-1])
            except ValueError:
                card_no = None
            if card_no is not None:
                cards, _ = detect_cards(page, layout)
                match = next((c for c in cards if c.occupied and c.index + 1 == card_no), None)
                if match is not None:
                    return _upscale(crop_card_padded(page, match), 2.0)
    image = cv2.imread(str(crop_path))
    if image is None:
        raise FileNotFoundError(crop_path)
    return _upscale(image, 2.0)


def retry_invalid_with_small(
    voters: list[VoterRecord],
    *,
    layout: LayoutProfile = DEFAULT_LAYOUT,
    profiler: PhaseProfiler | None = None,
    project_root: Path | None = None,
) -> tuple[list[VoterRecord], int, int]:
    """Re-OCR invalid cards with PP-OCRv6_small. Returns (voters, attempted, fixed)."""
    targets = [
        (i, rec)
        for i, rec in enumerate(voters)
        if not rec.valid and _resolve_retry_crop(rec.crop_path, project_root) is not None
    ]
    if not targets:
        return voters, 0, 0

    logger.info("retrying %s invalid card(s) with OCR model=small", len(targets))
    with _maybe_track(profiler, "OCR small retry init"):
        small = PaddleOcrEngine(model="small", profiler=profiler)
        small._ensure()

    fixed = 0
    with _maybe_track(profiler, "OCR small retry", count=len(targets)):
        for idx, rec in targets:
            crop_path = _resolve_retry_crop(rec.crop_path, project_root)
            assert crop_path is not None
            try:
                ocr_image = _retry_ocr_image(crop_path, rec, layout, project_root)
            except FileNotFoundError:
                logger.warning("retry skip unreadable crop %s", crop_path)
                continue
            result = small.recognize(ocr_image)
            meta = PageMeta(
                part_no=rec.part_no,
                constituency=rec.constituency,
                section=rec.section,
            )
            updated = parse_voter_card(
                result.tokens,
                page=rec.page,
                meta=meta,
                crop_width=ocr_image.shape[1],
                crop_height=ocr_image.shape[0],
                crop_path=rec.crop_path,
                mean_confidence=result.mean_confidence,
            )
            updated = merge_card_fields(updated, parse_labeled_lines(updated.raw_ocr))
            if needs_band_refill(updated):
                updated = refill_missing_fields(
                    updated, small, ocr_image, layout, profiler=profiler
                )
            updated = validate_voter(updated, layout)
            if updated.valid and not rec.valid:
                fixed += 1
            voters[idx] = updated

    logger.info("small retry fixed %s / %s invalid", fixed, len(targets))
    return voters, len(targets), fixed


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
    stats: dict | None = None,
    profile: bool = False,
    ocr_workers: int = 1,
    ocr_model: str = DEFAULT_OCR_MODEL,
    retry_small: bool = True,
    merge_keep: list[VoterRecord] | None = None,
    seed_extra: dict | None = None,
) -> list[VoterRecord]:
    paths.ensure()
    pdf_path = pdf_path.expanduser().resolve()
    profiler = PhaseProfiler() if profile else None
    ocr_workers = max(1, ocr_workers)
    ocr_model = (ocr_model or DEFAULT_OCR_MODEL).lower()

    logger.info("Opening %s", pdf_path)
    with _maybe_track(profiler, "PDF open"):
        doc = pymupdf.open(pdf_path)
    wanted = parse_page_spec(pages, doc.page_count)
    if wanted is not None:
        logger.info("OCR subset pages=%s", sorted(wanted))

    with _maybe_track(profiler, "OCR engine init"):
        if engine is not None:
            ocr = engine
            if hasattr(ocr, "set_profiler"):
                ocr.set_profiler(profiler)  # type: ignore[attr-defined]
            if hasattr(ocr, "_ensure"):
                ocr._ensure()  # type: ignore[attr-defined]
        elif ocr_workers == 1:
            ocr = PaddleOcrEngine(lang="en", profiler=profiler, model=ocr_model)
            ocr._ensure()
        else:
            ocr = OcrEnginePool(
                ocr_workers, lang="en", model=ocr_model, profiler=profiler
            )
            logger.info("OCR engine pool size=%s model=%s", len(ocr), ocr_model)

    voters: list[VoterRecord] = []
    global_meta = PageMeta()
    page_types: dict[int, str] = {}
    refill_calls = 0
    refill_skips = 0
    header_ocr_done = False
    small_retry_attempted = 0
    small_retry_fixed = 0

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
            with _maybe_track(profiler, "PDF render"):
                image = render_page(
                    doc, page_index, dpi=dpi, scale=scale, prefer_native=prefer_native
                )
            with _maybe_track(profiler, "Debug page write"):
                page_debug = paths.debug_dir / "pages" / f"page{page_no}.png"
                save_debug_image(image, page_debug)

            with _maybe_track(profiler, "Page classify"):
                page_type = classify_page(image, layout)
            page_types[page_no] = page_type.value
            logger.info("page %s → %s", page_no, page_type.value)

            if page_type == PageType.HEADER:
                cover = cv2.resize(image, (0, 0), fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
                with _maybe_track(profiler, "OCR cover"):
                    cover_tokens = ocr.recognize(cover).tokens
                with _maybe_track(profiler, "Parse cover"):
                    global_meta = merge_meta(global_meta, parse_cover_totals(cover_tokens))

            if page_type != PageType.VOTER:
                progress.advance(task)
                continue

            with _maybe_track(profiler, "Grid / card detect"):
                cards, method = detect_cards(image, layout)
            occupied = [c for c in cards if c.occupied]
            logger.info("page %s cards=%s method=%s", page_no, len(occupied), method.value)
            if visualize:
                with _maybe_track(profiler, "Debug overlay"):
                    vis = draw_detections(image, cards)
                    save_debug_image(vis, paths.debug_dir / f"page{page_no}_detected.png")
            if len(occupied) < layout.min_full_page_cards:
                save_debug_image(image, paths.debug_dir / f"page{page_no}_low_count.png")

            h, w = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            with _maybe_track(profiler, "Content frame"):
                frame = detect_content_frame(gray, layout)
            header_bottom = max(24, min(frame.y - 2, occupied[0].box.y - 2) if occupied else frame.y - 2)
            footer_top = min(h - 16, frame.y2 + 2)

            if not header_ocr_done:
                header_img = image[0:header_bottom, :]
                footer_img = image[footer_top:h, :]
                with _maybe_track(profiler, "OCR header/footer"):
                    header_tokens = ocr.recognize(_upscale(header_img, 2.0)).tokens
                    footer_tokens = ocr.recognize(_upscale(footer_img, 2.0)).tokens
                with _maybe_track(profiler, "Parse header"):
                    page_meta = parse_page_header(header_tokens + footer_tokens, page_no)
                    global_meta = merge_meta(global_meta, page_meta)
                header_ocr_done = True
            else:
                with _maybe_track(profiler, "Header cache reuse"):
                    page_meta = PageMeta(page=page_no)
            page_meta.constituency = page_meta.constituency or global_meta.constituency
            page_meta.part_no = page_meta.part_no or global_meta.part_no
            page_meta.section = page_meta.section or global_meta.section
            page_meta.constituency_no = page_meta.constituency_no or global_meta.constituency_no
            page_meta.section_no = page_meta.section_no or global_meta.section_no

            prepared: list[tuple] = []
            # Sparse / last voter pages: photo wipe often destroys EPIC; use full card.
            sparse_page = len(occupied) < layout.min_full_page_cards
            with _maybe_track(profiler, "Crop + upscale", count=len(occupied)):
                for card in occupied:
                    crop_name = f"page{page_no}_card{card.index + 1}.png"
                    crop_path = paths.debug_dir / "cards" / crop_name
                    save_card_crop(image, card, crop_path)
                    if sparse_page:
                        ocr_src = crop_card_padded(image, card)
                    else:
                        ocr_src = crop_card_for_ocr(image, card, layout)
                    ocr_image = _upscale(ocr_src, 2.0)
                    prepared.append((card, crop_path, ocr_image))

            ocr_results: list[OcrResult] = []
            with _maybe_track(profiler, "OCR card", count=len(prepared)):
                if ocr_workers == 1 or len(prepared) <= 1:
                    ocr_results = [ocr.recognize(img) for _, _, img in prepared]
                else:
                    with ThreadPoolExecutor(max_workers=ocr_workers) as pool:
                        ocr_results = list(
                            pool.map(lambda item: ocr.recognize(item[2]), prepared)
                        )

            for (card, crop_path, ocr_image), result in zip(prepared, ocr_results):
                with _maybe_track(profiler, "Parse card fields", count=1):
                    record = parse_voter_card(
                        result.tokens,
                        page=page_no,
                        meta=page_meta,
                        crop_width=ocr_image.shape[1],
                        crop_height=ocr_image.shape[0],
                        crop_path=str(crop_path),
                        mean_confidence=result.mean_confidence,
                    )
                    record = merge_card_fields(record, parse_labeled_lines(record.raw_ocr))
                if needs_band_refill(record):
                    with _maybe_track(profiler, "Band detect + upscale", count=1):
                        if sparse_page:
                            full_2x = _upscale(crop_card_padded(image, card), 2.0)
                        else:
                            full_2x = _upscale(crop_card(image, card), 2.0)
                    record = refill_missing_fields(
                        record, ocr, full_2x, layout, profiler=profiler
                    )
                    if record.age is None or record.gender is None:
                        record = refill_age_from_expanded_crop(
                            record, ocr, image, card.box, layout, profiler=profiler
                        )
                    refill_calls += 1
                else:
                    refill_skips += 1
                with _maybe_track(profiler, "Validate", count=1):
                    record = validate_voter(record, layout)
                if record.epic is None and record.name is None:
                    logger.debug("skip empty cell page=%s card=%s", page_no, card.index + 1)
                    continue
                voters.append(record)
            progress.advance(task)

    do_retry = retry_small and ocr_model != "small" and any(not v.valid for v in voters)
    if do_retry:
        voters, small_retry_attempted, small_retry_fixed = retry_invalid_with_small(
            voters,
            layout=layout,
            profiler=profiler,
            project_root=Path.cwd(),
        )

    if merge_keep is not None and wanted is not None:
        kept = [rec for rec in merge_keep if rec.page not in wanted]
        voters = kept + voters

    voters.sort(key=lambda v: (v.serial_no is None, v.serial_no or 0, v.page))
    seed = seed_extra or {}
    extra = {
        "source": seed.get("source") or str(pdf_path),
        "layout": layout.name,
        "pageTypes": {**(seed.get("pageTypes") or {}), **page_types},
        "partNo": global_meta.part_no or seed.get("partNo"),
        "constituency": global_meta.constituency or seed.get("constituency"),
        "expectedTotal": global_meta.expected_total
        if global_meta.expected_total is not None
        else seed.get("expectedTotal"),
        "extractedTotal": len(voters),
        "validCount": sum(1 for v in voters if v.valid),
        "refillCalls": refill_calls,
        "refillSkips": refill_skips,
        "ocrWorkers": ocr_workers,
        "ocrModel": ocr_model,
        "smallRetryAttempted": small_retry_attempted,
        "smallRetryFixed": small_retry_fixed,
    }
    if wanted is not None:
        extra["ocrPages"] = sorted(wanted)
    with _maybe_track(profiler, "Export JSON/CSV"):
        export_voters(
            voters,
            paths.json_dir / "voters.json",
            paths.csv_dir / "voters.csv",
            extra=extra,
        )
    if stats is not None:
        stats.update(extra)
        if profiler is not None:
            stats["profile"] = profiler.to_dict()
    logger.info(
        "extracted %s voters (%s valid) expected=%s refill=%s skip=%s "
        "model=%s small_retry=%s/%s",
        len(voters),
        extra["validCount"],
        global_meta.expected_total,
        refill_calls,
        refill_skips,
        ocr_model,
        small_retry_fixed,
        small_retry_attempted,
    )
    if profiler is not None:
        profiler.print_table(title=f"Parse timing — {pdf_path.name}", console=_profile_console)
        profile_path = paths.logs_dir / f"profile_{pdf_path.stem}.json"
        profiler.write_json(
            profile_path,
            extra={
                "source": str(pdf_path),
                "pages": pages,
                "extractedTotal": len(voters),
                "validCount": extra["validCount"],
                "refillCalls": refill_calls,
                "refillSkips": refill_skips,
                "ocrWorkers": ocr_workers,
                "ocrModel": ocr_model,
                "smallRetryAttempted": small_retry_attempted,
                "smallRetryFixed": small_retry_fixed,
            },
        )
    return voters


def retry_invalid_pages(
    pdf_path: Path,
    paths: RunPaths,
    *,
    json_path: Path | None = None,
    delete_pdf: bool = False,
    visualize: bool = False,
    dpi: int = 300,
    scale: float | None = None,
    prefer_native: bool = NATIVE_EXTRACT,
    layout: LayoutProfile = DEFAULT_LAYOUT,
    stats: dict | None = None,
    profile: bool = False,
    ocr_workers: int = 1,
    ocr_model: str = DEFAULT_OCR_MODEL,
    retry_small: bool = True,
) -> list[VoterRecord]:
    """Re-OCR only pages that still have invalid voters; merge back into voters.json."""
    paths.ensure()
    json_path = json_path or (paths.json_dir / "voters.json")
    if not json_path.is_file():
        raise FileNotFoundError(f"No existing parse at {json_path}; run parse first")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    existing = [_record_from_export(row) for row in (payload.get("voters") or [])]
    bad_pages = sorted({rec.page for rec in existing if not rec.valid})
    if not bad_pages:
        logger.info("no invalid voters in %s — nothing to retry", json_path)
        if stats is not None:
            stats.update(
                {
                    "extractedTotal": len(existing),
                    "validCount": sum(1 for v in existing if v.valid),
                    "expectedTotal": payload.get("expectedTotal"),
                    "retriedPages": [],
                }
            )
        return existing

    page_spec = ",".join(str(p) for p in bad_pages)
    logger.info(
        "retry-invalid %s pages=%s (%s invalid voters)",
        pdf_path.name,
        page_spec,
        sum(1 for v in existing if not v.valid),
    )
    page_stats: dict = {}
    refreshed = run_pipeline(
        pdf_path,
        paths,
        visualize=visualize,
        pages=page_spec,
        dpi=dpi,
        scale=scale,
        prefer_native=prefer_native,
        layout=layout,
        stats=page_stats,
        profile=profile,
        ocr_workers=ocr_workers,
        ocr_model=ocr_model,
        retry_small=retry_small,
        merge_keep=existing,
        seed_extra=payload,
    )
    if stats is not None:
        stats.update(page_stats)
        stats["retriedPages"] = bad_pages
    return refreshed


@contextmanager
def _maybe_track(profiler: PhaseProfiler | None, name: str, *, count: int = 1):
    if profiler is None:
        yield
    else:
        with profiler.track(name, count=count):
            yield
