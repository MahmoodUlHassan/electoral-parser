from __future__ import annotations

import fcntl
import json
import logging
import multiprocessing as mp
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from parser.logutil import setup_logging
from parser.coverage import backfill_from_output, mark_scanned, roll_rel_path
from parser.pipeline import RunPaths, refill_saved_json, retry_invalid_pages, run_pipeline
from parser.raster_retry import retry_invalid_from_rasters
from exporters.voters_db import (
    count_voters,
    db_path_for_combined,
    default_db_path,
    ensure_db_from_csv,
    import_csv,
    search as db_search,
    upsert_part_from_dataframe,
)
from parser.search import SEARCH_COLUMNS, load_voters, search_voters
from parser.smoke_compare import compare_page, format_stats, scan_pdf_page_assign

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True)

_PART_NUM = re.compile(r"part[_\-]?(\d+)", re.I)
logger = logging.getLogger("electoral.cli")

# Keep numeric columns stable across parts so concat/append never fights inference.
_CSV_OVERRIDES: dict[str, pl.DataType] = {
    "serialNo": pl.Int64,
    "age": pl.Int64,
    "page": pl.Int64,
    "partNo": pl.Int64,
}

def run_paths_for_pdf(pdf: Path, out_dir: Path, debug_dir: Path, log_dir: Path) -> RunPaths:
    """Nest under district/AC/part_N when the PDF lives in the downloads layout."""
    rel = roll_rel_path(pdf)
    return RunPaths(
        json_dir=out_dir / rel / "json",
        csv_dir=out_dir / rel / "csv",
        debug_dir=debug_dir / rel,
        logs_dir=log_dir,
    )



def _part_dirs_with_voters(root: Path) -> list[Path]:
    """part_* dirs (or a single part root) that already have voters.json."""
    root = root.expanduser().resolve()
    if (root / "json" / "voters.json").is_file():
        return [root]
    return sorted(
        p for p in root.glob("part_*") if (p / "json" / "voters.json").is_file()
    )


def _ac_rel_from_path(path: Path) -> Path | None:
    """District/AC relative path from a downloads or output AC/part path."""
    path = path.expanduser().resolve()
    probe = path if path.is_file() else path / "part_1.pdf"
    rel = roll_rel_path(probe)
    if len(rel.parts) >= 2 and rel.name.startswith("part_"):
        return rel.parent
    return None


def _resolve_raster_part_dirs(target: Path, out_dir: Path) -> list[Path]:
    """Locate parsed part dirs for raster retry when PDFs are gone.

    Accepts:
    - output/.../AC or output/.../AC/part_N
    - downloads/.../AC (empty of PDFs) → maps to out_dir/District/AC
    """
    target = target.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    parts = _part_dirs_with_voters(target)
    if parts:
        return parts
    ac_rel = _ac_rel_from_path(target)
    if ac_rel is not None:
        return _part_dirs_with_voters(out_dir / ac_rel)
    return []


def _run_raster_retry_parts(
    part_dirs: list[Path],
    *,
    out_dir: Path,
    debug_dir: Path,
    log_dir: Path,
    combined: Path,
    logger_,
    ocr_workers: int,
    ocr_model: str,
    ocr_mode: str,
    retry_small: bool,
    skip_age_gender_clip: bool,
) -> tuple[int, int]:
    """Shared body for raster-based invalid retry. Returns (parts_touched, valid_delta)."""
    out_dir = out_dir.expanduser().resolve()
    debug_dir = debug_dir.expanduser().resolve()
    total_delta = 0
    parts_touched = 0
    for part_dir in part_dirs:
        try:
            rel = part_dir.relative_to(out_dir)
        except ValueError:
            rel = Path(*part_dir.parts[-3:]) if len(part_dir.parts) >= 3 else Path(part_dir.name)
        paths = RunPaths(
            json_dir=out_dir / rel / "json",
            csv_dir=out_dir / rel / "csv",
            debug_dir=debug_dir / rel,
            logs_dir=log_dir,
        )
        stats: dict = {}
        try:
            retry_invalid_from_rasters(
                paths,
                stats=stats,
                ocr_workers=ocr_workers,
                ocr_model=ocr_model,
                retry_small=retry_small,
                ocr_mode=ocr_mode,
                skip_age_gender_clip=skip_age_gender_clip,
            )
        except Exception as exc:
            logger_.exception("raster-retry failed %s: %s", part_dir.name, exc)
            console.print(f"[red]{part_dir.name}: {exc}[/red]")
            continue
        pages = stats.get("retriedPages") or []
        if not pages:
            continue
        parts_touched += 1
        delta = int(stats.get("validDelta") or 0)
        total_delta += delta
        logger_.info(
            "raster-retry %s pages=%s valid %s→%s (Δ%+d) clip_skipped=%s",
            part_dir.name,
            pages,
            stats.get("validBefore"),
            stats.get("validAfter"),
            delta,
            stats.get("clipSkippedInvalids"),
        )
        console.print(
            f"[bold]{part_dir.name}[/bold] pages={pages} "
            f"valid {stats.get('validBefore')}→{stats.get('validAfter')} "
            f"(Δ{delta:+d}) clip_skipped={stats.get('clipSkippedInvalids')}"
        )
        _append_combined(paths.csv_dir / "voters.csv", f"{rel}.pdf", combined)
    return parts_touched, total_delta


def _sort_pdfs(pdfs: list[Path]) -> list[Path]:
    def key(p: Path) -> tuple[int, str]:
        m = _PART_NUM.search(p.stem)
        return (int(m.group(1)) if m else 10**9, p.name.lower())

    return sorted(pdfs, key=key)


def _read_voters_csv(path: Path) -> pl.DataFrame:
    """Read a voters CSV with stable dtypes (avoids Polars infer fighting across parts)."""
    return pl.read_csv(
        path,
        infer_schema_length=10_000,
        schema_overrides=_CSV_OVERRIDES,
        ignore_errors=True,
        truncate_ragged_lines=True,
    )


def _combined_sources_path(combined: Path) -> Path:
    return combined.with_suffix(combined.suffix + ".sources")


def _combined_source_name(pdf: Path) -> str:
    """Stable unique key for all_voters.csv (district/AC/part_N.pdf)."""
    return f"{roll_rel_path(pdf).as_posix()}.pdf"


def _append_combined(csv_path: Path, source_name: str, combined: Path) -> None:
    """Append one part's voters.csv into the combined ledger + SQLite search DB.

    Uses an exclusive lock + row append (not full rewrite) so parallel parses and
    large ledgers cannot corrupt all_voters.csv with partial overwrites/null bytes.
    A sidecar `.sources` list prevents duplicate CSV appends on parse resume/skip.

    SQLite (``csv/voters.db``) always replaces rows for ``source_name`` so
    retries/refills refresh UI search even when CSV append is skipped.
    """
    if not csv_path.exists():
        return
    df = _read_voters_csv(csv_path)
    if df.is_empty():
        return
    df = df.with_columns(pl.lit(source_name).alias("sourcePdf"))

    combined.parent.mkdir(parents=True, exist_ok=True)
    lock_path = combined.with_suffix(combined.suffix + ".lock")
    sources_path = _combined_sources_path(combined)
    db_path = db_path_for_combined(combined)
    with lock_path.open("a+", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            known: set[str] = set()
            if sources_path.is_file():
                known = {
                    ln.strip()
                    for ln in sources_path.read_text(encoding="utf-8").splitlines()
                    if ln.strip()
                }
            if source_name not in known:
                if not combined.exists() or combined.stat().st_size == 0:
                    df.write_csv(combined)
                else:
                    with combined.open("rb+") as fh:
                        fh.seek(0, 2)
                        if fh.tell() > 0:
                            fh.seek(-1, 2)
                            if fh.read(1) != b"\n":
                                fh.write(b"\n")
                    with combined.open("a", encoding="utf-8", newline="") as out:
                        df.write_csv(out, include_header=False)
                with sources_path.open("a", encoding="utf-8") as sf:
                    sf.write(source_name + "\n")
            # Always refresh search DB for this source (retry/refill-safe).
            upsert_part_from_dataframe(db_path, source_name, df)
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)






def parse_success(stats: dict) -> bool:
    """Enough valid rows to trust the parse before deleting the PDF."""
    valid = int(stats.get("validCount") or 0)
    extracted = int(stats.get("extractedTotal") or 0)
    expected = stats.get("expectedTotal")
    if valid <= 0 or extracted <= 0:
        return False
    if expected is not None and int(expected) > 0:
        return valid >= int(int(expected) * 0.90)
    return valid >= max(1, int(extracted * 0.90))


def _resolve_crop_file(crop_path: str | None, debug_dir: Path) -> Path | None:
    if not crop_path:
        return None
    candidate = Path(crop_path)
    if candidate.is_file():
        return candidate
    nested = Path.cwd() / crop_path
    if nested.is_file():
        return nested
    # Relative name under this part's cards dir
    name = Path(crop_path).name
    under = debug_dir / "cards" / name
    return under if under.is_file() else None


def cleanup_after_parse(
    pdf_path: Path,
    debug_dir: Path,
    *,
    delete_pdf: bool,
    voters: list | None = None,
) -> None:
    """Keep invalid card crops + their page rasters; drop the rest; optionally delete PDF."""
    invalid_pages: set[int] = set()
    if voters is not None:
        for rec in voters:
            if not getattr(rec, "valid", False):
                page = getattr(rec, "page", None)
                if page is not None:
                    invalid_pages.add(int(page))

    cards_dir = debug_dir / "cards"
    if voters is not None and cards_dir.is_dir():
        keep: set[Path] = set()
        removed_valid = 0
        for rec in voters:
            path = _resolve_crop_file(getattr(rec, "crop_path", None), debug_dir)
            if path is None:
                continue
            if getattr(rec, "valid", False):
                path.unlink(missing_ok=True)
                removed_valid += 1
            else:
                keep.add(path.resolve())
        for path in list(cards_dir.glob("*.png")):
            if path.resolve() not in keep:
                path.unlink(missing_ok=True)
        if cards_dir.is_dir() and not any(cards_dir.iterdir()):
            cards_dir.rmdir()
        logger.info(
            "card crops: removed_valid=%s kept_invalid=%s (%s)",
            removed_valid,
            len(keep),
            cards_dir,
        )
    elif cards_dir.is_dir():
        shutil.rmtree(cards_dir, ignore_errors=True)
        logger.info("removed card crops %s", cards_dir)

    pages_dir = debug_dir / "pages"
    kept_pages = 0
    removed_pages = 0
    if pages_dir.is_dir():
        for path in list(pages_dir.glob("page*.png")):
            # page37.png → 37
            stem = path.stem  # page37
            try:
                page_no = int(stem.removeprefix("page"))
            except ValueError:
                path.unlink(missing_ok=True)
                removed_pages += 1
                continue
            if page_no in invalid_pages:
                kept_pages += 1
            else:
                path.unlink(missing_ok=True)
                removed_pages += 1
        if pages_dir.is_dir() and not any(pages_dir.iterdir()):
            pages_dir.rmdir()
        logger.info(
            "page rasters: removed=%s kept_for_invalid_pages=%s pages=%s",
            removed_pages,
            kept_pages,
            sorted(invalid_pages) if invalid_pages else [],
        )

    for pattern in ("*_detected.png", "*_low_count.png"):
        for path in debug_dir.glob(pattern):
            # page37_low_count.png / page37_detected.png
            name = path.name
            keep_overlay = False
            for page_no in invalid_pages:
                if name.startswith(f"page{page_no}_"):
                    keep_overlay = True
                    break
            if not keep_overlay:
                path.unlink(missing_ok=True)

    if delete_pdf and pdf_path.is_file():
        pdf_path.unlink()
        logger.info("removed PDF %s", pdf_path)


def _parse_one_job(job: dict) -> dict:
    """Process entrypoint for --workers > 1 (spawn)."""
    item = Path(job["pdf"])
    stem_paths = RunPaths(
        json_dir=Path(job["json_dir"]),
        csv_dir=Path(job["csv_dir"]),
        debug_dir=Path(job["debug_dir"]),
        logs_dir=Path(job["logs_dir"]),
    )
    stats: dict = {}
    try:
        voters = run_pipeline(
            item,
            stem_paths,
            visualize=job["visualize"],
            pages=job["pages"],
            dpi=job["dpi"],
            scale=job["scale"],
            prefer_native=job["native"],
            stats=stats,
            profile=job.get("profile", False),
            ocr_workers=int(job.get("ocr_workers") or 1),
            ocr_model=str(job.get("ocr_model") or "tiny"),
            retry_small=bool(job.get("retry_small", True)),
            ocr_mode=str(job.get("ocr_mode") or "card"),
        )
        ok = parse_success(stats)
        if ok:
            mark_scanned(Path(job["out_dir"]), pdf_path=item, stats=stats, ok=True)
            cleanup_after_parse(
                item,
                stem_paths.debug_dir,
                delete_pdf=job["delete_pdf"],
                voters=voters,
            )
        return {
            "name": item.name,
            "source": _combined_source_name(item),
            "csv": str(stem_paths.csv_dir / "voters.csv"),
            "ok": ok,
            "stats": stats,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — surface to parent process
        return {
            "name": item.name,
            "source": _combined_source_name(item),
            "csv": str(stem_paths.csv_dir / "voters.csv"),
            "ok": False,
            "stats": stats,
            "error": str(exc),
        }


def _format_parse_stats(stats: dict | None) -> str:
    s = stats or {}
    return (
        f"expected={s.get('expectedTotal')} "
        f"extracted={s.get('extractedTotal')} "
        f"valid={s.get('validCount')} "
        f"refill={s.get('refillCalls')}/{s.get('refillSkips')} "
        f"ocr_workers={s.get('ocrWorkers')} "
        f"model={s.get('ocrModel')} "
        f"mode={s.get('ocrMode')} "
        f"small_retry={s.get('smallRetryFixed')}/{s.get('smallRetryAttempted')}"
    )


@app.command("parse")
def parse_cmd(
    pdf: Path = typer.Argument(..., exists=True, readable=True, help="PDF file or folder of part_*.pdf"),
    out_dir: Path = typer.Option(Path("output"), help="Root for json/ and csv/"),
    debug_dir: Path = typer.Option(Path("debug"), help="Crops, overlays, page rasters"),
    log_dir: Path = typer.Option(Path("logs"), help="Run logs"),
    visualize: bool = typer.Option(False, help="Draw numbered boxes (heavy; keep off for batches)"),
    pages: str | None = typer.Option(None, help="Subset like 3,14,22-30"),
    limit: int | None = typer.Option(None, help="Only the first N PDFs in a folder (use 1 to test)"),
    dpi: int = typer.Option(300),
    scale: float | None = typer.Option(None, help="PyMuPDF scale. Default: native raster."),
    native: bool = typer.Option(True),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    delete_pdf: bool = typer.Option(
        True,
        "--delete-pdf/--keep-pdf",
        help="After a successful parse, delete the source PDF and card crops",
    ),
    workers: int = typer.Option(1, help="Parallel PDFs (2–3). Each process loads its own Paddle."),
    profile: bool = typer.Option(
        False,
        "--profile/--no-profile",
        help="Print phase timing table and write logs/profile_<stem>.json (off by default)",
    ),
    ocr_workers: int = typer.Option(
        1,
        help="Parallel card OCR threads (pool of Paddle engines). CPU default 1; raise only if wall time drops.",
    ),
    ocr_model: str = typer.Option(
        "tiny",
        help="Primary OCR model: tiny | small | medium. Default tiny.",
    ),
    retry_small: bool = typer.Option(
        True,
        "--retry-small/--no-retry-small",
        help="After the PDF, re-OCR invalid cards with small (skipped if primary is already small).",
    ),
    ocr_mode: str = typer.Option(
        "card",
        "--ocr-mode",
        help="OCR strategy: card (per-card, default) | row (10 row strips + x-assign + recover).",
    ),
) -> None:
    """OCR electoral-roll PDFs into json/csv. Point at one file or the AC folder."""
    ocr_mode = ocr_mode.lower().strip()
    if ocr_mode not in ("card", "row"):
        raise typer.BadParameter("--ocr-mode must be card or row")
    logger_ = setup_logging(log_dir / "parse.log", verbose=verbose)
    combined = out_dir / "csv" / "all_voters.csv"
    workers = max(1, workers)

    if pdf.is_dir():
        pdfs = _sort_pdfs(list(pdf.glob("*.pdf")))
        if not pdfs:
            raise typer.BadParameter(f"No PDFs in {pdf}")
        if limit is not None:
            pdfs = pdfs[:limit]
        logger_.info("Parsing %s PDF(s) from %s workers=%s", len(pdfs), pdf, workers)

        jobs: list[dict] = []
        for item in pdfs:
            stem_paths = run_paths_for_pdf(item, out_dir, debug_dir, log_dir)
            skip = stem_paths.csv_dir / "voters.csv"
            if skip.exists() and skip.stat().st_size > 50:
                logger_.info("skip existing %s", skip)
                _append_combined(skip, _combined_source_name(item), combined)
                jpath = stem_paths.json_dir / "voters.json"
                skip_stats: dict = {}
                if jpath.is_file():
                    try:
                        payload = json.loads(jpath.read_text(encoding="utf-8"))
                        skip_stats = {
                            "validCount": payload.get("validCount"),
                            "extractedTotal": payload.get("extractedTotal"),
                            "expectedTotal": payload.get("expectedTotal"),
                            "ocrMode": payload.get("ocrMode"),
                        }
                    except (OSError, json.JSONDecodeError):
                        pass
                mark_scanned(out_dir, pdf_path=item, stats=skip_stats, ok=True)
                continue
            jobs.append(
                {
                    "pdf": str(item.resolve()),
                    "out_dir": str(out_dir.resolve()),
                    "json_dir": str(stem_paths.json_dir),
                    "csv_dir": str(stem_paths.csv_dir),
                    "debug_dir": str(stem_paths.debug_dir),
                    "logs_dir": str(stem_paths.logs_dir),
                    "visualize": visualize,
                    "pages": pages,
                    "dpi": dpi,
                    "scale": scale,
                    "native": native,
                    "delete_pdf": delete_pdf,
                    "profile": profile,
                    "ocr_workers": ocr_workers,
                    "ocr_model": ocr_model,
                    "retry_small": retry_small,
                    "ocr_mode": ocr_mode,
                }
            )

        if workers == 1:
            for job in jobs:
                logger_.info("=== %s ===", Path(job["pdf"]).name)
                result = _parse_one_job(job)
                if result["error"]:
                    console.print(f"[red]{result['name']}: {result['error']}[/red]")
                    raise typer.Exit(code=1)
                logger_.info(
                    "done %s ok=%s %s",
                    result["name"],
                    result["ok"],
                    _format_parse_stats(result.get("stats")),
                )
                _append_combined(Path(result["csv"]), result.get("source") or result["name"], combined)
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                futures = {pool.submit(_parse_one_job, job): job for job in jobs}
                for fut in as_completed(futures):
                    result = fut.result()
                    if result["error"]:
                        console.print(f"[red]{result['name']}: {result['error']}[/red]")
                        continue
                    logger_.info(
                        "done %s ok=%s %s",
                        result["name"],
                        result["ok"],
                        _format_parse_stats(result.get("stats")),
                    )
                    _append_combined(Path(result["csv"]), result.get("source") or result["name"], combined)

        console.print(f"[bold]Combined CSV[/bold]: {combined}")
        return

    # Nested: output/<District>/<NN_AC>/part_N/… and debug/<District>/<NN_AC>/part_N/
    paths = run_paths_for_pdf(pdf, out_dir, debug_dir, log_dir)
    stats: dict = {}
    voters = run_pipeline(
        pdf,
        paths,
        visualize=visualize,
        pages=pages,
        dpi=dpi,
        scale=scale,
        prefer_native=native,
        stats=stats,
        profile=profile,
        ocr_workers=ocr_workers,
        ocr_model=ocr_model,
        retry_small=retry_small,
        ocr_mode=ocr_mode,
    )
    _append_combined(paths.csv_dir / "voters.csv", _combined_source_name(pdf), combined)
    logger_.info("done %s ok=%s %s", pdf.name, parse_success(stats), _format_parse_stats(stats))
    if parse_success(stats):
        mark_scanned(out_dir, pdf_path=pdf.resolve(), stats=stats, ok=True)
        cleanup_after_parse(
            pdf.resolve(),
            paths.debug_dir,
            delete_pdf=delete_pdf,
            voters=voters,
        )
    else:
        logger_.warning(
            "keeping PDF (parse below success threshold): valid=%s expected=%s",
            stats.get("validCount"),
            stats.get("expectedTotal"),
        )


@app.command("smoke-compare")
def smoke_compare_cmd(
    pdf: Path = typer.Argument(..., exists=True, readable=True, help="One part PDF"),
    pages: str = typer.Option(..., help="Pages to compare, e.g. 4,32"),
    ocr_model: str = typer.Option("tiny"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    log_dir: Path = typer.Option(Path("logs")),
) -> None:
    """Side-by-side: per-card OCR @2× vs page OCR → bbox assign (experiment)."""
    setup_logging(log_dir / "smoke_compare.log", verbose=verbose)
    page_nums = sorted({int(p.strip()) for p in pages.split(",") if p.strip()})
    if not page_nums:
        raise typer.BadParameter("need at least one page")

    table = Table(title=f"smoke-compare {pdf.name}")
    table.add_column("page")
    table.add_column("path")
    table.add_column("sec", justify="right")
    table.add_column("ocr", justify="right")
    table.add_column("scale", justify="right")
    table.add_column("valid", justify="right")
    table.add_column("miss epic/age/house")

    for page_no in page_nums:
        console.print(f"[bold]page {page_no}[/bold]")
        card_s, page_s, row_s = compare_page(pdf, page_no, ocr_model=ocr_model)
        for s in (card_s, page_s, row_s):
            console.print(f"  {format_stats(s)}")
            table.add_row(
                str(page_no),
                s.name,
                f"{s.seconds:.2f}",
                str(s.ocr_calls),
                f"{s.scale:g}",
                f"{s.valid}/{s.extracted}",
                f"{s.missing_epic}/{s.missing_age}/{s.missing_house}",
            )
        if page_s.seconds > 0:
            console.print(
                f"  vs card: page={card_s.seconds / page_s.seconds:.2f}x  "
                f"row={card_s.seconds / row_s.seconds:.2f}x  "
                f"valid card/page/row={card_s.valid}/{page_s.valid}/{row_s.valid}"
            )
    console.print(table)


@app.command("smoke-page")
def smoke_page_cmd(
    pdf: Path = typer.Argument(..., exists=True, readable=True, help="One part PDF"),
    ocr_model: str = typer.Option("tiny"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    log_dir: Path = typer.Option(Path("logs")),
) -> None:
    """Full-PDF experiment: page OCR → assign on every voter page (no card×30)."""
    setup_logging(log_dir / "smoke_page.log", verbose=verbose)
    t0 = time.perf_counter()
    results = scan_pdf_page_assign(pdf, ocr_model=ocr_model)
    wall = time.perf_counter() - t0
    valid = sum(s.valid for s in results)
    extracted = sum(s.extracted for s in results)
    miss_e = sum(s.missing_epic for s in results)
    miss_a = sum(s.missing_age for s in results)
    miss_h = sum(s.missing_house for s in results)
    ocr_sum = sum(s.seconds for s in results)
    console.print(
        f"[bold]{pdf.name}[/bold] voter_pages={len(results)} wall={wall:.1f}s "
        f"(sum page OCR {ocr_sum:.1f}s) valid={valid}/{extracted} "
        f"miss epic/age/house={miss_e}/{miss_a}/{miss_h}"
    )
    table = Table(title=f"smoke-page {pdf.name}")
    table.add_column("page", justify="right")
    table.add_column("sec", justify="right")
    table.add_column("scale", justify="right")
    table.add_column("valid", justify="right")
    table.add_column("miss e/a/h")
    for s in results:
        table.add_row(
            str(s.page),
            f"{s.seconds:.2f}",
            f"{s.scale:g}",
            f"{s.valid}/{s.extracted}",
            f"{s.missing_epic}/{s.missing_age}/{s.missing_house}",
        )
    console.print(table)


@app.command("retry-invalid")
def retry_invalid_cmd(
    pdf: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="PDF file/folder, or AC folder (downloads or output) when PDFs are gone",
    ),
    out_dir: Path = typer.Option(Path("output"), help="Where parse wrote json/csv"),
    debug_dir: Path = typer.Option(Path("debug"), help="Crops / page rasters"),
    log_dir: Path = typer.Option(Path("logs"), help="Run logs"),
    visualize: bool = typer.Option(False),
    dpi: int = typer.Option(300),
    scale: float | None = typer.Option(None),
    native: bool = typer.Option(True),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    keep_pdf: bool = typer.Option(True, "--keep-pdf/--delete-pdf", help="Keep PDFs after retry"),
    ocr_workers: int = typer.Option(1),
    ocr_model: str = typer.Option("tiny"),
    ocr_mode: str = typer.Option("card", help="card | row (PDF retry and debug-PNG fallback)"),
    retry_small: bool = typer.Option(True, "--retry-small/--no-retry-small"),
) -> None:
    """Re-OCR only pages that still have invalid voters; merge into existing voters.json.

    If the PDF path has no PDFs, falls back to debug page PNGs under --debug-dir
    (same as retry-invalid-rasters), using matching parts under --out-dir.
    """
    logger_ = setup_logging(log_dir / "retry_invalid.log", verbose=verbose)
    combined = out_dir / "csv" / "all_voters.csv"
    pdfs = _sort_pdfs(list(pdf.glob("*.pdf"))) if pdf.is_dir() else ([pdf] if pdf.suffix.lower() == ".pdf" else [])
    if not pdfs:
        part_dirs = _resolve_raster_part_dirs(pdf, out_dir)
        if not part_dirs:
            raise typer.BadParameter(
                f"No PDFs in {pdf} and no part_*/json/voters.json found under "
                f"{out_dir} for that AC (need debug page PNGs for raster fallback)"
            )
        logger_.info(
            "retry-invalid: no PDFs in %s — falling back to debug page PNGs (%s parts)",
            pdf,
            len(part_dirs),
        )
        console.print(
            f"[yellow]No PDFs in {pdf}[/yellow] — falling back to debug page PNGs "
            f"({len(part_dirs)} parts)"
        )
        parts_touched, total_delta = _run_raster_retry_parts(
            part_dirs,
            out_dir=out_dir,
            debug_dir=debug_dir,
            log_dir=log_dir,
            combined=combined,
            logger_=logger_,
            ocr_workers=ocr_workers,
            ocr_model=ocr_model,
            ocr_mode=ocr_mode,
            retry_small=retry_small,
            skip_age_gender_clip=False,
        )
        console.print(
            f"[bold]Done[/bold] (raster fallback) parts_touched={parts_touched} "
            f"valid_delta={total_delta:+d} combined={combined}"
        )
        return

    for item in pdfs:
        stem_paths = run_paths_for_pdf(item, out_dir, debug_dir, log_dir)
        json_path = stem_paths.json_dir / "voters.json"
        if not json_path.is_file():
            logger_.warning("skip %s — no %s", item.name, json_path)
            continue
        stats: dict = {}
        try:
            voters = retry_invalid_pages(
                item,
                stem_paths,
                json_path=json_path,
                visualize=visualize,
                dpi=dpi,
                scale=scale,
                prefer_native=native,
                stats=stats,
                ocr_workers=ocr_workers,
                ocr_model=ocr_model,
                retry_small=retry_small,
                ocr_mode=ocr_mode,
            )
        except Exception as exc:
            logger_.exception("retry-invalid failed %s: %s", item.name, exc)
            console.print(f"[red]{item.name}: {exc}[/red]")
            continue
        if not (stats.get("retriedPages") or stats.get("ocrPages")):
            logger_.info("skip %s — no invalid pages", item.name)
            continue
        logger_.info(
            "retry-invalid %s pages=%s valid=%s/%s expected=%s",
            item.name,
            stats.get("retriedPages") or stats.get("ocrPages"),
            stats.get("validCount"),
            stats.get("extractedTotal"),
            stats.get("expectedTotal"),
        )
        console.print(
            f"[bold]{item.name}[/bold] pages={stats.get('retriedPages') or stats.get('ocrPages')} "
            f"valid={stats.get('validCount')}/{stats.get('extractedTotal')} "
            f"expected={stats.get('expectedTotal')}"
        )
        _append_combined(stem_paths.csv_dir / "voters.csv", _combined_source_name(item), combined)
        if parse_success(stats):
            mark_scanned(out_dir, pdf_path=item.resolve(), stats=stats, ok=True)
            cleanup_after_parse(
                item.resolve(),
                stem_paths.debug_dir,
                delete_pdf=not keep_pdf,
                voters=voters,
            )
    console.print(f"[bold]Combined CSV[/bold]: {combined}")


@app.command("retry-invalid-rasters")
def retry_invalid_rasters_cmd(
    target: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="AC folder under output/ (e.g. output/Rangareddy/51_Rajendranagar) or one part_* dir",
    ),
    out_dir: Path = typer.Option(Path("output"), help="Parse output root"),
    debug_dir: Path = typer.Option(Path("debug"), help="Debug root (page PNGs)"),
    log_dir: Path = typer.Option(Path("logs"), help="Run logs"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    ocr_workers: int = typer.Option(1),
    ocr_model: str = typer.Option("tiny"),
    ocr_mode: str = typer.Option("card", help="card | row"),
    retry_small: bool = typer.Option(True, "--retry-small/--no-retry-small"),
    skip_age_gender_clip: bool = typer.Option(
        True,
        "--skip-age-gender-clip/--include-age-gender-clip",
        help="Skip voters whose only errors are exactly missing_age+missing_gender (6-line wrap).",
    ),
) -> None:
    """Re-OCR non-clip invalid pages from debug page PNGs (no PDF required)."""
    logger_ = setup_logging(log_dir / "retry_invalid_rasters.log", verbose=verbose)
    combined = out_dir / "csv" / "all_voters.csv"
    out_dir = out_dir.expanduser().resolve()
    debug_dir = debug_dir.expanduser().resolve()
    target = target.expanduser().resolve()

    part_dirs = _resolve_raster_part_dirs(target, out_dir)
    if not part_dirs:
        raise typer.BadParameter(f"No part_*/json/voters.json under {target}")

    logger_.info(
        "raster-retry target=%s parts=%s skip_clip=%s mode=%s model=%s",
        target,
        len(part_dirs),
        skip_age_gender_clip,
        ocr_mode,
        ocr_model,
    )
    parts_touched, total_delta = _run_raster_retry_parts(
        part_dirs,
        out_dir=out_dir,
        debug_dir=debug_dir,
        log_dir=log_dir,
        combined=combined,
        logger_=logger_,
        ocr_workers=ocr_workers,
        ocr_model=ocr_model,
        ocr_mode=ocr_mode,
        retry_small=retry_small,
        skip_age_gender_clip=skip_age_gender_clip,
    )
    console.print(
        f"[bold]Done[/bold] parts_touched={parts_touched} valid_delta={total_delta:+d} "
        f"combined={combined}"
    )


@app.command("coverage-backfill")
def coverage_backfill_cmd(
    out_dir: Path = typer.Option(Path("output"), help="Parse output root (coverage.json lives here)"),
) -> None:
    """Rebuild coverage.json from existing voters.json source paths."""
    stats = backfill_from_output(out_dir)
    console.print(
        f"coverage ledger {out_dir / 'coverage.json'}: "
        f"added={stats['added']} updated={stats['updated']} skipped={stats['skipped']}"
    )


@app.command("refill")
def refill_cmd(
    out_dir: Path = typer.Option(Path("output"), help="Where parse wrote json/csv"),
    json_path: Path | None = typer.Option(None, "--json", help="One voters.json; default: all under out_dir"),
) -> None:
    """Fill missing house/age from saved card crops. Skips a full PDF re-OCR."""
    if json_path is not None:
        targets = [json_path]
    else:
        nested = sorted(out_dir.glob("**/part_*/json/voters.json"))
        root_json = out_dir / "json" / "voters.json"
        targets = nested if nested else ([root_json] if root_json.exists() else [])
    if not targets:
        raise typer.BadParameter(f"No voters.json under {out_dir}. Run parse first.")
    combined = out_dir / "csv" / "all_voters.csv"
    for path in targets:
        stats = refill_saved_json(path)
        console.print(
            f"{path}: house +{stats['houseFilled']} age +{stats['ageFilled']} "
            f"valid={stats['valid']}/{stats['total']} missing_crop={stats['missingCrop']}"
        )
        csv_file = path.parent.parent / "csv" / "voters.csv"
        if not csv_file.exists():
            continue
        parent = path.parent.parent
        if parent.resolve() == out_dir.resolve():
            source_name = Path(
                json.loads(path.read_text(encoding="utf-8")).get("source") or "roll.pdf"
            ).name
        else:
            source_name = f"{parent.name}.pdf"
        _append_combined(csv_file, source_name, combined)
    console.print(f"[bold]Combined CSV[/bold]: {combined}")


@app.command("db-import")
def db_import_cmd(
    out_dir: Path = typer.Option(Path("output"), help="Where parse wrote csv/"),
    csv: Path | None = typer.Option(
        None, help="CSV to import; default is out_dir/csv/all_voters.csv"
    ),
    db: Path | None = typer.Option(
        None, help="SQLite path; default is out_dir/csv/voters.db"
    ),
) -> None:
    """One-shot import of all_voters.csv into voters.db for UI search."""
    csv_path = Path(csv) if csv else out_dir / "csv" / "all_voters.csv"
    db_path = Path(db) if db else default_db_path(out_dir)
    if not csv_path.is_file():
        raise typer.BadParameter(f"No CSV at {csv_path}")
    console.print(f"Importing {csv_path} → {db_path} …")
    n = import_csv(csv_path, db_path, replace=True)
    console.print(f"[bold]Done[/bold] rows={n} db={db_path}")


@app.command("search")
def search_cmd(
    name: str = typer.Argument(..., help="Substring match on name or relativeName"),
    out_dir: Path = typer.Option(Path("output"), help="Where parse wrote csv/"),
    csv: Path | None = typer.Option(
        None,
        help="Force CSV search (loads full file). Default: query voters.db",
    ),
) -> None:
    """Find voters by name after parse. Case-insensitive substring (SQLite)."""
    needle = name.strip()
    if needle == "":
        raise typer.BadParameter("Empty name")

    if csv is not None:
        df = load_voters(out_dir, csv)
        if df.is_empty():
            raise typer.BadParameter("CSV files are empty.")
        hits_df = search_voters(df, needle)
        if hits_df.is_empty():
            console.print(f"No matches for [bold]{name}[/bold]")
            raise typer.Exit(code=1)
        cols = [c for c in SEARCH_COLUMNS if c in hits_df.columns]
        table = Table(title=f"{hits_df.height} match(es) for “{name}”")
        for c in cols:
            table.add_column(c)
        for row in hits_df.select(cols).iter_rows():
            table.add_row(*["" if v is None else str(v) for v in row])
        console.print(table)
        return

    db_path = default_db_path(out_dir)
    ensure_db_from_csv(out_dir)
    if count_voters(db_path) == 0:
        raise typer.BadParameter(
            f"No voters in {db_path}. Run parse or: python main.py db-import"
        )
    rows = db_search(db_path, needle, limit=200)
    if not rows:
        console.print(f"No matches for [bold]{name}[/bold]")
        raise typer.Exit(code=1)

    table = Table(title=f"{len(rows)} match(es) for “{name}” (showing up to 200)")
    for c in SEARCH_COLUMNS:
        table.add_column(c)
    for row in rows:
        table.add_row(*["" if row.get(c) is None else str(row.get(c)) for c in SEARCH_COLUMNS])
    console.print(table)


@app.command("ui")
def ui_cmd(
    out_dir: Path = typer.Option(Path("output"), help="Where parse wrote csv/"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8765),
    backfill: bool = typer.Option(
        True,
        "--backfill/--no-backfill",
        help="Refresh coverage.json from voters.json sources before serving",
    ),
) -> None:
    """Open a local search page for parsed voters (reads csv/voters.db)."""
    from ui.server import serve

    csv = out_dir / "csv" / "all_voters.csv"
    db_path = default_db_path(out_dir)
    if not csv.exists() and count_voters(db_path) == 0:
        raise typer.BadParameter(
            f"No combined CSV at {csv} and no voters.db. Run parse first."
        )
    if backfill:
        stats = backfill_from_output(out_dir)
        console.print(
            f"coverage backfill: added={stats['added']} updated={stats['updated']} "
            f"skipped={stats['skipped']}"
        )
    if csv.exists():
        console.print(f"Ensuring search DB at {db_path} (imports CSV if empty)…")
        ensure_db_from_csv(out_dir)
    console.print(f"voters.db rows={count_voters(db_path)}")
    serve(out_dir.resolve(), host=host, port=port)


if __name__ == "__main__":
    app()
