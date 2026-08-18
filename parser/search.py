from __future__ import annotations

from pathlib import Path

import polars as pl

from exporters.writers import CSV_COLUMNS

SEARCH_COLUMNS = CSV_COLUMNS + ["sourcePdf"]
TEXT_FIELDS = ("name", "relativeName", "epic", "houseNo", "section", "sourcePdf")


def load_voters(out_dir: Path, csv: Path | None = None) -> pl.DataFrame:
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


def rows_as_dicts(df: pl.DataFrame) -> list[dict]:
    if df.is_empty():
        return []
    keep = [c for c in SEARCH_COLUMNS if c in df.columns]
    return df.select(keep).to_dicts()
