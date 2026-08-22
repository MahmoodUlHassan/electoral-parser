from __future__ import annotations

import json
import logging
import multiprocessing as mp
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from parser.logutil import setup_logging
from parser.pipeline import RunPaths, refill_saved_json, retry_invalid_pages, run_pipeline
from parser.search import SEARCH_COLUMNS, load_voters, search_voters

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True)

_PART_NUM = re.compile(r"part[_\-]?(\d+)", re.I)
logger = logging.getLogger("electoral.cli")


def _sort_pdfs(pdfs: list[Path]) -> list[Path]:
    def key(p: Path) -> tuple[int, str]:
        m = _PART_NUM.search(p.stem)
        return (int(m.group(1)) if m else 10**9, p.name.lower())

    return sorted(pdfs, key=key)


def _append_combined(csv_path: Path, source_name: str, combined: Path) -> None:
    if not csv_path.exists():
        return
    df = pl.read_csv(csv_path)
    if df.is_empty():
        return
    df = df.with_columns(pl.lit(source_name).alias("sourcePdf"))
    if combined.exists():
        prev = pl.read_csv(combined)
        # Drop older rows from this same PDF on re-run.
        if "sourcePdf" in prev.columns:
            prev = prev.filter(pl.col("sourcePdf") != source_name)
        df = pl.concat([prev, df], how="diagonal_relaxed")
    combined.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(combined)


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
        )
        ok = parse_success(stats)
        if ok:
            cleanup_after_parse(
                item,
                stem_paths.debug_dir,
                delete_pdf=job["delete_pdf"],
                voters=voters,
            )
        return {
            "name": item.name,
            "csv": str(stem_paths.csv_dir / "voters.csv"),
            "ok": ok,
            "stats": stats,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — surface to parent process
        return {
            "name": item.name,
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
) -> None:
    """OCR electoral-roll PDFs into json/csv. Point at one file or the AC folder."""
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
            stem_paths = RunPaths(
                json_dir=out_dir / item.stem / "json",
                csv_dir=out_dir / item.stem / "csv",
                debug_dir=debug_dir / item.stem,
                logs_dir=log_dir,
            )
            skip = stem_paths.csv_dir / "voters.csv"
            if skip.exists() and skip.stat().st_size > 50:
                logger_.info("skip existing %s", skip)
                _append_combined(skip, item.name, combined)
                continue
            jobs.append(
                {
                    "pdf": str(item.resolve()),
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
                _append_combined(Path(result["csv"]), result["name"], combined)
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
                    _append_combined(Path(result["csv"]), result["name"], combined)

        console.print(f"[bold]Combined CSV[/bold]: {combined}")
        return

    # Same nesting as folder mode: output/<stem>/… and debug/<stem>/cards/
    stem = pdf.stem
    paths = RunPaths(
        json_dir=out_dir / stem / "json",
        csv_dir=out_dir / stem / "csv",
        debug_dir=debug_dir / stem,
        logs_dir=log_dir,
    )
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
    )
    _append_combined(paths.csv_dir / "voters.csv", pdf.name, combined)
    logger_.info("done %s ok=%s %s", pdf.name, parse_success(stats), _format_parse_stats(stats))
    if parse_success(stats):
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


@app.command("retry-invalid")
def retry_invalid_cmd(
    pdf: Path = typer.Argument(..., exists=True, readable=True, help="PDF file or folder of part_*.pdf"),
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
    retry_small: bool = typer.Option(True, "--retry-small/--no-retry-small"),
) -> None:
    """Re-OCR only pages that still have invalid voters; merge into existing voters.json."""
    logger_ = setup_logging(log_dir / "retry_invalid.log", verbose=verbose)
    combined = out_dir / "csv" / "all_voters.csv"
    pdfs = _sort_pdfs(list(pdf.glob("*.pdf"))) if pdf.is_dir() else [pdf]
    if not pdfs:
        raise typer.BadParameter(f"No PDFs in {pdf}")

    for item in pdfs:
        stem_paths = RunPaths(
            json_dir=out_dir / item.stem / "json",
            csv_dir=out_dir / item.stem / "csv",
            debug_dir=debug_dir / item.stem,
            logs_dir=log_dir,
        )
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
        _append_combined(stem_paths.csv_dir / "voters.csv", item.name, combined)
        if parse_success(stats):
            cleanup_after_parse(
                item.resolve(),
                stem_paths.debug_dir,
                delete_pdf=not keep_pdf,
                voters=voters,
            )
    console.print(f"[bold]Combined CSV[/bold]: {combined}")


@app.command("refill")
def refill_cmd(
    out_dir: Path = typer.Option(Path("output"), help="Where parse wrote json/csv"),
    json_path: Path | None = typer.Option(None, "--json", help="One voters.json; default: all under out_dir"),
) -> None:
    """Fill missing house/age from saved card crops. Skips a full PDF re-OCR."""
    if json_path is not None:
        targets = [json_path]
    else:
        nested = sorted(out_dir.glob("*/json/voters.json"))
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


@app.command("search")
def search_cmd(
    name: str = typer.Argument(..., help="Substring match on name or relativeName"),
    out_dir: Path = typer.Option(Path("output"), help="Where parse wrote csv/"),
    csv: Path | None = typer.Option(None, help="Explicit voters.csv; default is output/csv/all_voters.csv plus globs"),
) -> None:
    """Find voters by name after parse. Case-insensitive substring."""
    df = load_voters(out_dir, csv)
    if df.is_empty():
        raise typer.BadParameter("CSV files are empty.")
    needle = name.strip()
    if needle == "":
        raise typer.BadParameter("Empty name")

    hits = search_voters(df, needle)

    if hits.is_empty():
        console.print(f"No matches for [bold]{name}[/bold]")
        raise typer.Exit(code=1)

    cols = [c for c in SEARCH_COLUMNS if c in hits.columns]
    table = Table(title=f"{hits.height} match(es) for “{name}”")
    for c in cols:
        table.add_column(c)
    for row in hits.select(cols).iter_rows():
        table.add_row(*["" if v is None else str(v) for v in row])
    console.print(table)


@app.command("ui")
def ui_cmd(
    out_dir: Path = typer.Option(Path("output"), help="Where parse wrote csv/"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8765),
) -> None:
    """Open a local search page for parsed voters."""
    from ui.server import serve

    csv = out_dir / "csv" / "all_voters.csv"
    if not csv.exists():
        raise typer.BadParameter(f"No combined CSV at {csv}. Run parse first.")
    serve(out_dir.resolve(), host=host, port=port)


if __name__ == "__main__":
    app()
