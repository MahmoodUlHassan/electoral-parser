"""Coverage ledger: districts → ACs → part numbers scanned vs pending."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("electoral.coverage")

PARTS_PER_AC = 700
LEDGER_NAME = "coverage.json"

_PART_RE = re.compile(r"^part_(\d+)$", re.I)
_AC_FOLDER_RE = re.compile(r"^(\d+)_(.+)$")


def default_metadata_paths() -> list[Path]:
    here = Path(__file__).resolve()
    root = here.parents[1]
    return [
        root / "metadata" / "telangana.json",
        root.parent / "electoral-downloader" / "src" / "metadata.json",
        Path.home() / "Documents/Projects/AI_Projects/electoral-downloader/src/metadata.json",
    ]


def load_metadata(path: Path | None = None) -> dict[str, Any]:
    candidates = [path] if path else default_metadata_paths()
    for candidate in candidates:
        if candidate is None:
            continue
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "Telangana metadata.json not found. Expected sibling electoral-downloader/src/metadata.json"
    )


def ledger_path(out_dir: Path) -> Path:
    return out_dir.expanduser().resolve() / LEDGER_NAME


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _key(district: str, asmbly_no: int, part_no: int) -> str:
    return f"{district}|{asmbly_no}|{part_no}"


def parse_roll_location(pdf_path: Path | str) -> dict[str, Any] | None:
    """Extract district / asmblyNo / partNo from a downloads/.../NN_Name/part_N.pdf path."""
    path = Path(pdf_path).expanduser()
    try:
        path = path.resolve()
    except OSError:
        path = Path(pdf_path)
    parts = path.parts
    stem_m = _PART_RE.match(path.stem)
    if not stem_m:
        return None
    part_no = int(stem_m.group(1))
    # …/District/52_Serilingampally/part_9.pdf
    if len(parts) < 3:
        return None
    ac_folder = parts[-2]
    district = parts[-3]
    ac_m = _AC_FOLDER_RE.match(ac_folder)
    if not ac_m:
        return None
    return {
        "district": district.replace("_", " "),
        "districtFolder": district,
        "asmblyNo": int(ac_m.group(1)),
        "acName": ac_m.group(2).replace("_", " "),
        "acFolder": ac_folder,
        "partNo": part_no,
        "source": str(path),
    }


def roll_rel_path(pdf_path: Path | str) -> Path:
    """Relative path district/AC/part_N for output and debug trees."""
    loc = parse_roll_location(pdf_path)
    if loc:
        return Path(loc["districtFolder"]) / loc["acFolder"] / f"part_{loc['partNo']}"
    stem = Path(pdf_path).stem
    return Path(stem)


def _empty_ledger(parts_per_ac: int = PARTS_PER_AC) -> dict[str, Any]:
    return {
        "version": 1,
        "partsPerAc": parts_per_ac,
        "updatedAt": _now(),
        "scanned": {},
    }


def read_ledger(out_dir: Path) -> dict[str, Any]:
    path = ledger_path(out_dir)
    if not path.is_file():
        return _empty_ledger()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("failed reading coverage ledger %s", path)
        return _empty_ledger()
    data.setdefault("version", 1)
    data.setdefault("partsPerAc", PARTS_PER_AC)
    data.setdefault("scanned", {})
    return data


def _write_ledger(out_dir: Path, data: dict[str, Any]) -> None:
    path = ledger_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updatedAt"] = _now()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def mark_scanned(
    out_dir: Path,
    *,
    pdf_path: Path | str | None = None,
    location: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
    ok: bool = True,
) -> bool:
    """Record a scanned part. Returns False if path could not be mapped to district/AC/part."""
    loc = location or (parse_roll_location(pdf_path) if pdf_path else None)
    if not loc:
        logger.debug("coverage: skip unmapped path %s", pdf_path)
        return False
    if not ok:
        return False

    out_dir = out_dir.expanduser().resolve()
    lock_path = out_dir / ".coverage.lock"
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)

    import fcntl

    with lock_path.open("a+", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            data = read_ledger(out_dir)
            key = _key(loc["districtFolder"], int(loc["asmblyNo"]), int(loc["partNo"]))
            stats = stats or {}
            entry = {
                "district": loc["district"],
                "districtFolder": loc["districtFolder"],
                "asmblyNo": int(loc["asmblyNo"]),
                "acName": loc.get("acName"),
                "partNo": int(loc["partNo"]),
                "source": loc.get("source") or (str(pdf_path) if pdf_path else None),
                "validCount": stats.get("validCount"),
                "extractedTotal": stats.get("extractedTotal"),
                "expectedTotal": stats.get("expectedTotal"),
                "ocrMode": stats.get("ocrMode"),
                "scannedAt": _now(),
            }
            data["scanned"][key] = entry
            _write_ledger(out_dir, data)
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    return True


def backfill_from_output(
    out_dir: Path,
    *,
    metadata_path: Path | None = None,
) -> dict[str, int]:
    """Scan output/*/json/voters.json source fields and merge into the ledger."""
    out_dir = out_dir.expanduser().resolve()
    added = updated = skipped = 0
    for json_path in sorted(out_dir.glob("**/part_*/json/voters.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue
        source = payload.get("source")
        loc = parse_roll_location(source) if source else None
        if not loc:
            skipped += 1
            continue
        stats = {
            "validCount": payload.get("validCount"),
            "extractedTotal": payload.get("extractedTotal"),
            "expectedTotal": payload.get("expectedTotal"),
            "ocrMode": payload.get("ocrMode"),
        }
        key = _key(loc["districtFolder"], int(loc["asmblyNo"]), int(loc["partNo"]))
        before = key in read_ledger(out_dir).get("scanned", {})
        mark_scanned(out_dir, location=loc, stats=stats, ok=True)
        if before:
            updated += 1
        else:
            added += 1
    # Touch metadata load so callers know catalog is available
    try:
        load_metadata(metadata_path)
    except FileNotFoundError:
        pass
    return {"added": added, "updated": updated, "skipped": skipped}


def default_downloads_roots() -> list[Path]:
    here = Path(__file__).resolve()
    root = here.parents[1]
    return [
        root.parent / "electoral-downloader" / "downloads",
        Path.home() / "Documents/Projects/AI_Projects/electoral-downloader/downloads",
    ]


def resolve_downloads_root(path: Path | None = None) -> Path | None:
    if path is not None and path.is_dir():
        return path.expanduser().resolve()
    for candidate in default_downloads_roots():
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _safe_path_segment(value: str) -> str | None:
    """Reject empty / traversal / separator segments for downloads paths."""
    s = (value or "").strip()
    if not s or s in (".", "..") or "/" in s or "\\" in s or "\x00" in s:
        return None
    return s


def ac_folder_name(asmbly_no: int, ac_name: str) -> str:
    return f"{int(asmbly_no)}_" + str(ac_name or "").replace(" ", "_")


def pdf_api_url(district_folder: str, ac_folder: str, part_no: int) -> str:
    return (
        "/api/pdf?district="
        + district_folder
        + "&ac="
        + ac_folder
        + "&part="
        + str(int(part_no))
    )


def resolve_part_pdf(
    district_folder: str,
    ac_folder: str,
    part_no: int,
    *,
    downloads_root: Path | None = None,
) -> Path | None:
    """Resolve downloads/<district>/<ac>/part_N.pdf under the downloads root only."""
    d = _safe_path_segment(district_folder.replace(" ", "_") if district_folder else "")
    ac = _safe_path_segment(ac_folder)
    if d is None or ac is None:
        return None
    try:
        n = int(part_no)
    except (TypeError, ValueError):
        return None
    if n < 1:
        return None
    root = resolve_downloads_root(downloads_root)
    if root is None:
        return None
    root = root.resolve()
    candidate = (root / d / ac / f"part_{n}.pdf").resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    if not _PART_RE.match(candidate.stem) or candidate.suffix.lower() != ".pdf":
        return None
    return candidate


def _invalid_count_from_voters_json(path: Path) -> int | None:
    """Return invalid voter count from voters.json summary or per-row flags."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    extracted = payload.get("extractedTotal")
    valid = payload.get("validCount")
    if extracted is not None and valid is not None:
        try:
            return max(0, int(extracted) - int(valid))
        except (TypeError, ValueError):
            pass
    voters = payload.get("voters")
    if not isinstance(voters, list):
        return None
    bad = 0
    for v in voters:
        if not isinstance(v, dict):
            continue
        errors = v.get("errors") or []
        if v.get("valid") is False or (isinstance(errors, list) and len(errors) > 0):
            bad += 1
    return bad


def _invalid_count_from_ledger_entry(entry: dict[str, Any] | None) -> int | None:
    if not entry:
        return None
    extracted = entry.get("extractedTotal")
    valid = entry.get("validCount")
    if extracted is None or valid is None:
        return None
    try:
        return max(0, int(extracted) - int(valid))
    except (TypeError, ValueError):
        return None


def ac_download_part_numbers(
    district_folder: str,
    asmbly_no: int,
    ac_name: str,
    *,
    downloads_root: Path | None = None,
) -> set[int]:
    """Part numbers present as part_*.pdf under downloads/<district>/<NN_Name>/."""
    root = resolve_downloads_root(downloads_root)
    if root is None:
        return set()
    district_dir = root / district_folder
    if not district_dir.is_dir():
        alt = root / district_folder.replace(" ", "_")
        if alt.is_dir():
            district_dir = alt
    if not district_dir.is_dir():
        return set()
    want = f"{int(asmbly_no)}_" + str(ac_name).replace(" ", "_")
    ac_dir = district_dir / want
    if not ac_dir.is_dir():
        matches = sorted(district_dir.glob(f"{int(asmbly_no)}_*"))
        if len(matches) != 1:
            return set()
        ac_dir = matches[0]
    parts: set[int] = set()
    for pdf in ac_dir.glob("part_*.pdf"):
        m = _PART_RE.match(pdf.stem)
        if m:
            parts.add(int(m.group(1)))
    return parts


def ac_part_total(
    scanned_parts: set[int] | list[int],
    *,
    downloaded_parts: set[int] | None = None,
    default: int = PARTS_PER_AC,
) -> int:
    """On-disk PDF count for the AC; else 700 if unscanned; else max scanned partNo."""
    downloaded = set(downloaded_parts or ())
    scanned = {int(p) for p in scanned_parts}
    if downloaded:
        return len(downloaded)
    if not scanned:
        return default
    return max(scanned)


def build_coverage_tree(
    out_dir: Path,
    *,
    metadata_path: Path | None = None,
    downloads_root: Path | None = None,
    parts_per_ac: int = PARTS_PER_AC,
) -> dict[str, Any]:
    meta = load_metadata(metadata_path)
    ledger = read_ledger(out_dir)
    scanned = ledger.get("scanned") or {}
    parts_per_ac = int(ledger.get("partsPerAc") or parts_per_ac)
    dl_root = resolve_downloads_root(downloads_root)

    by_ac: dict[tuple[str, int], set[int]] = {}
    for entry in scanned.values():
        folder = entry.get("districtFolder") or str(entry.get("district") or "").replace(" ", "_")
        try:
            ac = int(entry["asmblyNo"])
            part = int(entry["partNo"])
        except (KeyError, TypeError, ValueError):
            continue
        by_ac.setdefault((folder, ac), set()).add(part)

    districts_out: list[dict[str, Any]] = []
    total_acs = 0
    total_scanned = 0
    grand_total = 0
    for dist in meta.get("districts") or []:
        d_name = dist.get("districtName") or ""
        d_folder = d_name.replace(" ", "_")
        assemblies_out: list[dict[str, Any]] = []
        d_scanned = 0
        d_total = 0
        for asm in dist.get("assemblies") or []:
            ac_no = int(asm["asmblyNo"])
            ac_name = str(asm.get("name") or "")
            parts = by_ac.get((d_folder, ac_no), set())
            if not parts:
                for (folder, no), pset in by_ac.items():
                    if no == ac_no and folder.replace("_", " ").lower() == d_name.lower():
                        parts = pset
                        break
            downloaded = ac_download_part_numbers(
                d_folder, ac_no, ac_name, downloads_root=dl_root
            )
            n = len(parts)
            total = ac_part_total(parts, downloaded_parts=downloaded, default=parts_per_ac)
            d_scanned += n
            d_total += total
            total_scanned += n
            total_acs += 1
            assemblies_out.append(
                {
                    "asmblyNo": ac_no,
                    "name": ac_name,
                    "scanned": n,
                    "total": total,
                    "pending": max(0, total - n),
                    "downloaded": len(downloaded),
                }
            )
        assemblies_out.sort(key=lambda a: a["asmblyNo"])
        grand_total += d_total
        districts_out.append(
            {
                "name": d_name,
                "districtCd": dist.get("districtCd"),
                "folder": d_folder,
                "scanned": d_scanned,
                "total": d_total,
                "pending": max(0, d_total - d_scanned),
                "assemblies": assemblies_out,
            }
        )

    districts_out.sort(key=lambda d: d["name"])
    return {
        "state": meta.get("state"),
        "stateName": meta.get("stateName"),
        "partsPerAc": parts_per_ac,
        "downloadsRoot": str(dl_root) if dl_root else None,
        "updatedAt": ledger.get("updatedAt"),
        "summary": {
            "districts": len(districts_out),
            "assemblies": total_acs,
            "scannedParts": total_scanned,
            "pendingParts": max(0, grand_total - total_scanned),
            "totalParts": grand_total,
        },
        "districts": districts_out,
    }


def ac_parts_payload(
    out_dir: Path,
    *,
    district: str,
    asmbly_no: int,
    metadata_path: Path | None = None,
    downloads_root: Path | None = None,
    parts_per_ac: int = PARTS_PER_AC,
) -> dict[str, Any]:
    out_dir = out_dir.expanduser().resolve()
    tree = build_coverage_tree(
        out_dir,
        metadata_path=metadata_path,
        downloads_root=downloads_root,
        parts_per_ac=parts_per_ac,
    )
    parts_per_ac = int(tree["partsPerAc"])
    ledger = read_ledger(out_dir)
    want_district = district.replace("_", " ").strip().lower()
    scanned_parts: list[int] = []
    ac_name = None
    ledger_by_part: dict[int, dict[str, Any]] = {}
    for entry in (ledger.get("scanned") or {}).values():
        try:
            if int(entry["asmblyNo"]) != int(asmbly_no):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        entry_district = str(
            entry.get("district") or entry.get("districtFolder") or ""
        ).replace("_", " ").strip().lower()
        if entry_district and entry_district != want_district:
            continue
        part_no = int(entry["partNo"])
        scanned_parts.append(part_no)
        ledger_by_part[part_no] = entry
        ac_name = entry.get("acName") or ac_name
    scanned_parts = sorted(set(scanned_parts))
    for dist in tree["districts"]:
        if dist["name"].lower() == want_district:
            for asm in dist["assemblies"]:
                if asm["asmblyNo"] == int(asmbly_no):
                    ac_name = asm["name"]
                    break
            break
    scanned_set = set(scanned_parts)
    d_folder = district.replace(" ", "_")
    ac_folder = ac_folder_name(int(asmbly_no), ac_name or "")
    downloaded = ac_download_part_numbers(
        d_folder, int(asmbly_no), ac_name or "", downloads_root=downloads_root
    )
    total = ac_part_total(scanned_set, downloaded_parts=downloaded, default=parts_per_ac)
    if downloaded:
        part_nos = sorted(downloaded | scanned_set)
    else:
        part_nos = list(range(1, total + 1))

    cells: list[dict[str, Any]] = []
    for n in part_nos:
        scanned = n in scanned_set
        invalid: int | None = None
        vj = out_dir / d_folder / ac_folder / f"part_{n}" / "json" / "voters.json"
        invalid = _invalid_count_from_voters_json(vj)
        if invalid is None:
            invalid = _invalid_count_from_ledger_entry(ledger_by_part.get(n))
        if invalid is None:
            invalid = 0
        pdf_path = resolve_part_pdf(
            d_folder, ac_folder, n, downloads_root=downloads_root
        )
        has_pdf = pdf_path is not None
        cell: dict[str, Any] = {
            "partNo": n,
            "scanned": scanned,
            "invalidCount": int(invalid or 0),
            "hasPdf": has_pdf,
        }
        if has_pdf:
            cell["pdfUrl"] = pdf_api_url(d_folder, ac_folder, n)
        cells.append(cell)

    return {
        "district": district.replace("_", " "),
        "districtFolder": d_folder,
        "asmblyNo": int(asmbly_no),
        "name": ac_name,
        "acFolder": ac_folder,
        "scanned": len(scanned_parts),
        "total": total,
        "downloaded": len(downloaded),
        "scannedParts": scanned_parts,
        "cells": cells,
    }
