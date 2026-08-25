"""SQLite ledger for UI search (parallel to all_voters.csv).

DB path: ``{out_dir}/csv/voters.db`` (same folder as ``all_voters.csv``).

Per-part CSV/JSON exports are unchanged. Combined CSV still append-once via
``.sources``; this DB **replaces** all rows for a ``sourcePdf`` on each merge
so retries/refills refresh search without rewriting the CSV.
"""

from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

from exporters.writers import CSV_COLUMNS

SEARCH_COLUMNS = CSV_COLUMNS + ["sourcePdf"]
TEXT_FIELDS = ("name", "relativeName", "epic", "houseNo", "section", "sourcePdf")

_AC_FOLDER_RE = re.compile(r"^(\d+)_(.+)$")
_BATCH = 5_000

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS voters (
    serialNo INTEGER,
    epic TEXT,
    name TEXT,
    relationType TEXT,
    relativeName TEXT,
    houseNo TEXT,
    age INTEGER,
    gender TEXT,
    page INTEGER,
    partNo INTEGER,
    constituency TEXT,
    section TEXT,
    sourcePdf TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_voters_source ON voters(sourcePdf);
CREATE INDEX IF NOT EXISTS idx_voters_part ON voters(partNo);
"""


def default_db_path(out_dir: Path) -> Path:
    """Canonical path next to the combined CSV."""
    return Path(out_dir) / "csv" / "voters.db"


def db_path_for_combined(combined: Path) -> Path:
    return Path(combined).parent / "voters.db"


def connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_CREATE_SQL)
    conn.commit()


def _as_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v)
    return s if s else None


def _row_tuple(row: dict[str, Any], source_pdf: str | None = None) -> tuple:
    sp = source_pdf if source_pdf is not None else row.get("sourcePdf")
    return (
        _as_int(row.get("serialNo")),
        _as_str(row.get("epic")),
        _as_str(row.get("name")),
        _as_str(row.get("relationType")),
        _as_str(row.get("relativeName")),
        _as_str(row.get("houseNo")),
        _as_int(row.get("age")),
        _as_str(row.get("gender")),
        _as_int(row.get("page")),
        _as_int(row.get("partNo")),
        _as_str(row.get("constituency")),
        _as_str(row.get("section")),
        _as_str(sp) or "",
    )


_INSERT_SQL = (
    "INSERT INTO voters ("
    + ", ".join(SEARCH_COLUMNS)
    + ") VALUES ("
    + ", ".join("?" for _ in SEARCH_COLUMNS)
    + ")"
)


def upsert_part(
    db_path: Path,
    source_pdf: str,
    rows: Sequence[dict[str, Any]] | Iterable[dict[str, Any]],
) -> int:
    """Delete all voters for ``source_pdf``, then insert ``rows``. Returns insert count."""
    rows_list = list(rows)
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("DELETE FROM voters WHERE sourcePdf = ?", (source_pdf,))
            if rows_list:
                payload = [_row_tuple(r, source_pdf) for r in rows_list]
                conn.executemany(_INSERT_SQL, payload)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return len(rows_list)


def upsert_part_from_dataframe(db_path: Path, source_pdf: str, df: Any) -> int:
    """Upsert from a Polars DataFrame (columns matching SEARCH_COLUMNS / CSV)."""
    if hasattr(df, "is_empty") and df.is_empty():
        return upsert_part(db_path, source_pdf, [])
    cols = [c for c in SEARCH_COLUMNS if c != "sourcePdf" and c in df.columns]
    # Ensure sourcePdf is set; other SEARCH_COLUMNS may be missing.
    records = df.select(cols).to_dicts() if cols else [{} for _ in range(df.height)]
    return upsert_part(db_path, source_pdf, records)


def count_voters(db_path: Path) -> int:
    if not Path(db_path).is_file():
        return 0
    with connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM voters").fetchone()
        return int(row["n"]) if row else 0


def db_is_empty(db_path: Path) -> bool:
    p = Path(db_path)
    if not p.is_file() or p.stat().st_size == 0:
        return True
    return count_voters(p) == 0


def _ac_constituency_clause(ac_folder: str) -> tuple[str, list[Any]] | None:
    """SQL + params matching legacy bare part_N.pdf rows via constituency."""
    m = _AC_FOLDER_RE.match(ac_folder)
    if not m:
        return None
    ac_name = m.group(2).replace("_", " ").strip().lower()
    ac_us = m.group(2).strip().lower()
    if not ac_name:
        return None
    sql = (
        "(LOWER(IFNULL(constituency, '')) LIKE ? "
        "OR LOWER(IFNULL(constituency, '')) LIKE ?)"
    )
    return sql, [f"%{ac_name}%", f"%{ac_us}%"]


def _coverage_where(
    *,
    district: str | None,
    ac: str | None,
    asmbly_no: int | None,
    part_no: int | None,
) -> tuple[str, list[Any]]:
    """Build AND-joined WHERE for coverage filters (mirrors filter_by_coverage)."""
    if not any([district, ac, asmbly_no is not None, part_no is not None]):
        return "", []

    clauses: list[str] = []
    params: list[Any] = []
    bare = "INSTR(IFNULL(sourcePdf, ''), '/') = 0"
    ac_folder = ac.strip().replace(" ", "_") if ac else None
    ac_clause = _ac_constituency_clause(ac_folder) if ac_folder else None

    if district:
        d_folder = district.strip().replace(" ", "_")
        d_space = district.strip().replace("_", " ")
        d_raw = district.strip()
        params.extend([f"{d_folder}/%", f"{d_space}/%", f"{d_raw}/%"])
        path_d = (
            "(IFNULL(sourcePdf, '') LIKE ? "
            "OR IFNULL(sourcePdf, '') LIKE ? "
            "OR IFNULL(sourcePdf, '') LIKE ?)"
        )
        if ac_clause:
            ac_sql, ac_params = ac_clause
            params.extend(ac_params)
            clauses.append(f"({path_d} OR ({bare} AND {ac_sql}))")
        else:
            clauses.append(path_d)

    if ac_folder:
        params.append(f"%/{ac_folder}/%")
        path_ac = "IFNULL(sourcePdf, '') LIKE ?"
        if ac_clause:
            ac_sql, ac_params = ac_clause
            params.extend(ac_params)
            clauses.append(f"({path_ac} OR {ac_sql})")
        else:
            clauses.append(path_ac)
    elif asmbly_no is not None:
        params.append(f"%/{int(asmbly_no)}_%")
        clauses.append("IFNULL(sourcePdf, '') LIKE ?")

    if part_no is not None:
        n = int(part_no)
        params.extend([f"%/part_{n}.pdf", f"part_{n}.pdf", n])
        clauses.append(
            "(IFNULL(sourcePdf, '') LIKE ? OR IFNULL(sourcePdf, '') = ? OR partNo = ?)"
        )

    return " AND ".join(clauses), params


def _text_where(query: str) -> tuple[str, list[Any]]:
    tokens = [t.lower() for t in query.split() if t.strip()]
    if not tokens:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    field_exprs = [f"LOWER(IFNULL({f}, ''))" for f in TEXT_FIELDS]
    blob = " || ' ' || ".join(field_exprs)
    for tok in tokens:
        params.append(f"%{tok}%")
        clauses.append(f"({blob}) LIKE ?")
    return " AND ".join(clauses), params


def search(
    db_path: Path,
    query: str = "",
    *,
    district: str | None = None,
    ac: str | None = None,
    asmbly_no: int | None = None,
    part_no: int | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Filter + text search without loading the full CSV. Returns up to ``limit`` rows."""
    q = (query or "").strip()
    has_filter = bool(district or ac or asmbly_no is not None or part_no is not None)
    if not q and not has_filter:
        return []
    if not Path(db_path).is_file():
        return []

    where_parts: list[str] = []
    params: list[Any] = []

    cov_sql, cov_params = _coverage_where(
        district=district, ac=ac, asmbly_no=asmbly_no, part_no=part_no
    )
    if cov_sql:
        where_parts.append(f"({cov_sql})")
        params.extend(cov_params)

    if q:
        text_sql, text_params = _text_where(q)
        if text_sql:
            where_parts.append(f"({text_sql})")
            params.extend(text_params)
        elif not has_filter:
            return []

    where = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
    sql = (
        f"SELECT {', '.join(SEARCH_COLUMNS)} FROM voters{where} LIMIT ?"
    )
    params.append(int(limit))

    with connect(db_path) as conn:
        cur = conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def search_count(
    db_path: Path,
    query: str = "",
    *,
    district: str | None = None,
    ac: str | None = None,
    asmbly_no: int | None = None,
    part_no: int | None = None,
) -> int:
    """Total matches (no limit) for API ``count`` field."""
    q = (query or "").strip()
    has_filter = bool(district or ac or asmbly_no is not None or part_no is not None)
    if not q and not has_filter:
        return 0
    if not Path(db_path).is_file():
        return 0

    where_parts: list[str] = []
    params: list[Any] = []
    cov_sql, cov_params = _coverage_where(
        district=district, ac=ac, asmbly_no=asmbly_no, part_no=part_no
    )
    if cov_sql:
        where_parts.append(f"({cov_sql})")
        params.extend(cov_params)
    if q:
        text_sql, text_params = _text_where(q)
        if text_sql:
            where_parts.append(f"({text_sql})")
            params.extend(text_params)
        elif not has_filter:
            return 0

    where = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
    with connect(db_path) as conn:
        row = conn.execute(f"SELECT COUNT(*) AS n FROM voters{where}", params).fetchone()
        return int(row["n"]) if row else 0


def import_csv(
    csv_path: Path,
    db_path: Path,
    *,
    replace: bool = True,
    batch_size: int = _BATCH,
) -> int:
    """Import ``all_voters.csv`` (or any voters CSV with sourcePdf) into SQLite.

    Uses one transaction + batched executemany. With ``replace=True`` (default),
    drops and recreates the table so a one-shot migrate is clean.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        # executescript auto-COMMITs; run DDL outside the insert transaction.
        if replace:
            conn.executescript("DROP TABLE IF EXISTS voters;\n" + _CREATE_SQL)
        conn.execute("BEGIN IMMEDIATE")
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                batch: list[tuple] = []
                for row in reader:
                    batch.append(_row_tuple(row))
                    if len(batch) >= batch_size:
                        conn.executemany(_INSERT_SQL, batch)
                        total += len(batch)
                        batch.clear()
                if batch:
                    conn.executemany(_INSERT_SQL, batch)
                    total += len(batch)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return total


def ensure_db_from_csv(out_dir: Path, *, csv_path: Path | None = None) -> Path:
    """If voters.db is missing/empty and combined CSV exists, import it. Returns db path."""
    out_dir = Path(out_dir)
    db = default_db_path(out_dir)
    combined = Path(csv_path) if csv_path else out_dir / "csv" / "all_voters.csv"
    if not db_is_empty(db):
        return db
    if combined.is_file() and combined.stat().st_size > 20:
        import_csv(combined, db, replace=True)
    return db
