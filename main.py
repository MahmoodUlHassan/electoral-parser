from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from parser.logutil import setup_logging
from parser.pipeline import RunPaths, refill_saved_json, run_pipeline
from parser.search import SEARCH_COLUMNS, load_voters, search_voters

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True)

_PART_NUM = re.compile(r"part[_\-]?(\d+)", re.I)


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
) -> None:
    """OCR electoral-roll PDFs into json/csv. Point at one file or the AC folder."""
    logger = setup_logging(log_dir / "parse.log", verbose=verbose)
    combined = out_dir / "csv" / "all_voters.csv"

    if pdf.is_dir():
        pdfs = _sort_pdfs(list(pdf.glob("*.pdf")))
        if not pdfs:
            raise typer.BadParameter(f"No PDFs in {pdf}")
        if limit is not None:
            pdfs = pdfs[:limit]
        logger.info("Parsing %s PDF(s) from %s", len(pdfs), pdf)
        for item in pdfs:
            logger.info("=== %s ===", item.name)
            stem_paths = RunPaths(
                json_dir=out_dir / item.stem / "json",
                csv_dir=out_dir / item.stem / "csv",
                debug_dir=debug_dir / item.stem,
                logs_dir=log_dir,
            )
            skip = stem_paths.csv_dir / "voters.csv"
            if skip.exists() and skip.stat().st_size > 50:
                logger.info("skip existing %s", skip)
                _append_combined(skip, item.name, combined)
                continue
            run_pipeline(
                item,
                stem_paths,
                visualize=visualize,
                pages=pages,
                dpi=dpi,
                scale=scale,
                prefer_native=native,
            )
            _append_combined(stem_paths.csv_dir / "voters.csv", item.name, combined)
        console.print(f"[bold]Combined CSV[/bold]: {combined}")
        return

    paths = RunPaths(
        json_dir=out_dir / "json",
        csv_dir=out_dir / "csv",
        debug_dir=debug_dir,
        logs_dir=log_dir,
    )
    run_pipeline(
        pdf,
        paths,
        visualize=visualize,
        pages=pages,
        dpi=dpi,
        scale=scale,
        prefer_native=native,
    )
    _append_combined(paths.csv_dir / "voters.csv", pdf.name, combined)


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
