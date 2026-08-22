from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from parser.models import VoterRecord

CSV_COLUMNS = [
    "serialNo",
    "epic",
    "name",
    "relationType",
    "relativeName",
    "houseNo",
    "age",
    "gender",
    "page",
    "partNo",
    "constituency",
    "section",
]

# Explicit schema — mixed null/str section (and similar) breaks Polars infer.
CSV_SCHEMA: dict[str, pl.DataType] = {
    "serialNo": pl.Int64,
    "epic": pl.Utf8,
    "name": pl.Utf8,
    "relationType": pl.Utf8,
    "relativeName": pl.Utf8,
    "houseNo": pl.Utf8,
    "age": pl.Int64,
    "gender": pl.Utf8,
    "page": pl.Int64,
    "partNo": pl.Int64,
    "constituency": pl.Utf8,
    "section": pl.Utf8,
}


def export_voters(
    voters: list[VoterRecord],
    json_path: Path,
    csv_path: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = extra.copy() if extra else {}
    payload["voters"] = [v.to_full_dict() for v in voters]
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [v.to_public_dict() for v in voters]
    if rows:
        df = pl.from_dicts(rows, schema=CSV_SCHEMA).select(CSV_COLUMNS)
    else:
        df = pl.DataFrame(schema=CSV_SCHEMA)
    df.write_csv(csv_path)
