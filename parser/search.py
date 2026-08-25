from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from exporters.writers import CSV_COLUMNS
from exporters.voters_db import (
    count_voters,
    default_db_path,
    ensure_db_from_csv,
    search as db_search,
    search_count as db_search_count,
)

SEARCH_COLUMNS = CSV_COLUMNS + ["sourcePdf"]
TEXT_FIELDS = ("name", "relativeName", "epic", "houseNo", "section", "sourcePdf")

_AC_FOLDER_RE = re.compile(r"^(\d+)_(.+)$")


def load_voters(out_dir: Path, csv: Path | None = None) -> pl.DataFrame:
    """Load voters from CSV (full file). Prefer query_voters / voters.db for UI."""
    files: list[Path] = []
    if csv is not None:
        files = [csv]
    else:
        combined = out_dir / "csv" / "all_voters.csv"
        files = [combined] if combined.exists() else list(out_dir.rglob("voters.csv"))
    frames = [pl.read_csv(p) for p in files if p.exists() and p.stat().st_size > 20]
    if not frames:
        return pl.DataFrame()
    df = pl.concat(frames, how="diagonal_relaxed")
    for col in SEARCH_COLUMNS:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.String).alias(col))
    return df


def search_voters(df: pl.DataFrame, query: str) -> pl.DataFrame:
    tokens = [t.lower() for t in query.split() if t.strip()]
    if not tokens or df.is_empty():
        return df.head(0)

    blob = pl.lit("")
    for field in TEXT_FIELDS:
        if field in df.columns:
            blob = blob + " " + pl.col(field).cast(pl.String).fill_null("")
    hay = blob.str.to_lowercase()
    mask = pl.lit(True)
    for tok in tokens:
        mask = mask & hay.str.contains(tok, literal=True)
    return df.filter(mask)


def _ac_constituency_mask(df: pl.DataFrame, ac_folder: str) -> pl.Expr:
    """Match legacy rows whose sourcePdf is bare part_N.pdf via constituency."""
    if "constituency" not in df.columns:
        return pl.lit(False)
    m = _AC_FOLDER_RE.match(ac_folder)
    if not m:
        return pl.lit(False)
    ac_name = m.group(2).replace("_", " ").strip().lower()
    ac_us = m.group(2).strip().lower()
    if not ac_name:
        return pl.lit(False)
    const = pl.col("constituency").cast(pl.String).fill_null("").str.to_lowercase()
    return const.str.contains(ac_name, literal=True) | const.str.contains(ac_us, literal=True)


def filter_by_coverage(
    df: pl.DataFrame,
    *,
    district: str | None = None,
    ac: str | None = None,
    asmbly_no: int | None = None,
    part_no: int | None = None,
) -> pl.DataFrame:
    """Filter voters by sourcePdf path: DistrictFolder/ACFolder/part_N.pdf.

    Legacy combined CSV rows may only have bare ``part_N.pdf`` in sourcePdf; those
    are matched via constituency + partNo when path prefixes are missing.
    """
    if df.is_empty():
        return df
    if not any([district, ac, asmbly_no is not None, part_no is not None]):
        return df
    if "sourcePdf" not in df.columns:
        return df.head(0)

    sp = pl.col("sourcePdf").cast(pl.String).fill_null("")
    bare = ~sp.str.contains("/", literal=True)
    mask = pl.lit(True)

    ac_folder = ac.strip().replace(" ", "_") if ac else None
    ac_field = _ac_constituency_mask(df, ac_folder) if ac_folder else pl.lit(False)

    if district:
        d_folder = district.strip().replace(" ", "_")
        d_space = district.strip().replace("_", " ")
        path_d = (
            sp.str.starts_with(d_folder + "/")
            | sp.str.starts_with(d_space + "/")
            | sp.str.starts_with(district.strip() + "/")
        )
        # Bare sourcePdf has no district segment; allow only when AC also matches.
        mask = mask & (path_d | (bare & ac_field))

    if ac_folder:
        path_ac = sp.str.contains("/" + ac_folder + "/", literal=True)
        mask = mask & (path_ac | ac_field)
    elif asmbly_no is not None:
        mask = mask & sp.str.contains(f"/{int(asmbly_no)}_", literal=True)

    if part_no is not None:
        n = int(part_no)
        part_mask = sp.str.ends_with(f"/part_{n}.pdf") | (sp == f"part_{n}.pdf")
        if "partNo" in df.columns:
            part_mask = part_mask | (pl.col("partNo").cast(pl.Int64, strict=False) == n)
        mask = mask & part_mask

    return df.filter(mask)


def rows_as_dicts(df: pl.DataFrame) -> list[dict]:
    if df.is_empty():
        return []
    keep = [c for c in SEARCH_COLUMNS if c in df.columns]
    return df.select(keep).to_dicts()


def query_voters(
    out_dir: Path,
    query: str = "",
    *,
    district: str | None = None,
    ac: str | None = None,
    asmbly_no: int | None = None,
    part_no: int | None = None,
    limit: int = 200,
) -> tuple[int, list[dict]]:
    """Search via SQLite (``out_dir/csv/voters.db``). Returns (total_count, rows)."""
    db_path = default_db_path(out_dir)
    ensure_db_from_csv(out_dir)
    total = db_search_count(
        db_path,
        query,
        district=district,
        ac=ac,
        asmbly_no=asmbly_no,
        part_no=part_no,
    )
    rows = db_search(
        db_path,
        query,
        district=district,
        ac=ac,
        asmbly_no=asmbly_no,
        part_no=part_no,
        limit=limit,
    )
    return total, rows


def voters_total(out_dir: Path) -> int:
    """Row count from voters.db (imports from CSV once if DB empty)."""
    ensure_db_from_csv(out_dir)
    return count_voters(default_db_path(out_dir))
