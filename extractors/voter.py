from __future__ import annotations

import re

from parser.models import OcrToken, PageMeta, VoterRecord

_EPIC_RE = re.compile(r"[A-Z]{3}[0-9]{7}")
# Tiny OCR often mangles Age/Gender on noisy last pages (Aae/Ace/Cender/Mele…).
_AGE_RE = re.compile(
    r"(?:Age|Aae|Aao|Ace|Aqe|A\.?\s*ae|A\s*ge)\s*[:\-·•.]?\s*([0-9OoIl]{1,3})",
    re.I,
)
_GENDER_RE = re.compile(
    r"(?:Gender|Cender|Condor|Gondor|Gend[eo]r|Ccnder)\s*[:\-·•.]?\s*"
    r"(Male|Female|Mele|Femele|Femcle|Femole|Eomolo|Molo|Femaie|Others?|Third\s*Gender)",
    re.I,
)
_HOUSE_RE = re.compile(r"Hou[a-z]{0,3}\s*Num[a-z]{0,4}\s*[:\-]?\s*(.+)$", re.I)
_HNO_RE = re.compile(r"H\.?\s*No\.?\s*[:\-]?\s*(.+)$", re.I)
_HOUSE_LABEL_RE = re.compile(r"^(?:Hou[a-z]{0,3}\s*Num[a-z]{0,4}|H\.?\s*No\.?)\s*[:\-]?$", re.I)
_FIELD_LEAK_RE = re.compile(r"name|age|gender|father|mother|husband|wife|photo|available", re.I)
_NAME_RE = re.compile(r"^(?:Name)\s*[:\-]\s*(.+)$", re.I)
_SERIAL_RE = re.compile(r"^(\d{1,4})$")

_RELATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Father'?s?\s*Name\s*[:\-]?\s*(.*)$", re.I), "Father"),
    (re.compile(r"Mother'?s?\s*Name\s*[:\-]?\s*(.*)$", re.I), "Mother"),
    (re.compile(r"Husband'?s?\s*Name\s*[:\-]?\s*(.*)$", re.I), "Husband"),
    (re.compile(r"Wife'?s?\s*Name\s*[:\-]?\s*(.*)$", re.I), "Wife"),
    (re.compile(r"^Others?\s*[:\-]\s*(.*)$", re.I), "Other"),
]


def group_lines(tokens: list[OcrToken], y_tol: float | None = None) -> list[str]:
    if not tokens:
        return []
    ordered = sorted(tokens, key=lambda t: (t.y0, t.x0))
    if y_tol is None:
        heights = [max(p[1] for p in t.bbox) - t.y0 for t in tokens]
        y_tol = max(8.0, float(sorted(heights)[len(heights) // 2]) * 0.6)
    lines: list[list[OcrToken]] = []
    for token in ordered:
        if lines and abs(token.cy - lines[-1][0].cy) <= y_tol:
            lines[-1].append(token)
        else:
            lines.append([token])
    joined: list[str] = []
    for line in lines:
        line.sort(key=lambda t: t.x0)
        joined.append(" ".join(t.text for t in line))
    return joined


def _clean_value(text: str) -> str:
    text = text.strip(" :-\t")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _title_person(name: str) -> str:
    name = _clean_value(name)
    # Keep existing mixed case from OCR; only title-case ALL CAPS.
    if name.isupper():
        return name.title()
    return name


def _normalize_epic(text: str) -> str | None:
    compact = re.sub(r"[^A-Z0-9]", "", text.upper())
    match = _EPIC_RE.search(compact)
    if match:
        return match.group(0)
    if len(compact) >= 10:
        candidate = compact[:10]
        letters = []
        for ch in candidate[:3]:
            letters.append("O" if ch == "0" else ("I" if ch == "1" else ch))
        digits = []
        for ch in candidate[3:]:
            if ch in "O":
                digits.append("0")
            elif ch in "IL":
                digits.append("1")
            else:
                digits.append(ch)
        fixed = "".join(letters) + "".join(digits)
        if _EPIC_RE.fullmatch(fixed):
            return fixed
    return None


def normalize_epic(text: str) -> str | None:
    """Public wrapper used by epic-strip refill."""
    return _normalize_epic(text)

def parse_house_line(text: str) -> str | None:
    blob = _clean_value(re.sub(r"\s+", " ", text or ""))
    if not blob or _HOUSE_LABEL_RE.match(blob):
        return None
    blob = re.split(r"\s+Age\s*[:\-]", blob, maxsplit=1, flags=re.I)[0]
    blob = _clean_value(blob)
    for pattern in (_HOUSE_RE, _HNO_RE):
        m = pattern.search(blob)
        if m:
            value = _clean_value(m.group(1))
            value = re.split(r"\s+Age\s*[:\-]", value, maxsplit=1, flags=re.I)[0]
            value = re.split(r"\s+[A-Za-z]{2,8}\s*[:\-]\s*\d{1,3}\b", value, maxsplit=1)[0]
            value = re.sub(r"\s+A$", "", _clean_value(value))
            return value or None
    if _FIELD_LEAK_RE.search(blob):
        return None
    if any(ch.isdigit() for ch in blob):
        return blob
    return None


def _normalize_age_digits(raw: str) -> int | None:
    digits = (
        raw.upper()
        .replace("O", "0")
        .replace("Ó", "0")
        .replace("Ò", "0")
        .replace("I", "1")
        .replace("L", "1")
    )
    digits = re.sub(r"\D", "", digits)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def parse_age_gender_line(text: str) -> tuple[int | None, str | None]:
    blob = re.sub(r"\s+", " ", text or "")
    age: int | None = None
    gender: str | None = None
    m = _AGE_RE.search(blob)
    if m:
        age = _normalize_age_digits(m.group(1))
    m = _GENDER_RE.search(blob)
    if m:
        gender = _normalize_gender(m.group(1))
    return age, gender


_SERIAL_COMBO_RE = re.compile(r"^\d{1,4}\s+[A-Z]{3}\d{7}$", re.I)


def _skip_field_line(line: str) -> bool:
    s = line.strip()
    if not s or re.search(r"photo|available", s, re.I):
        return True
    if _SERIAL_RE.fullmatch(s) or _SERIAL_COMBO_RE.fullmatch(s):
        return True
    return False


def _is_wrap_junk(line: str) -> bool:
    s = _clean_value(line)
    if not s or _SERIAL_RE.fullmatch(s) or _SERIAL_COMBO_RE.fullmatch(s):
        return True
    if re.fullmatch(r"[\d./\-]+", s):
        return True
    return False


def _strip_trailing_serial(value: str | None) -> str | None:
    if not value:
        return value
    cleaned = re.sub(r"\s+\d{1,4}$", "", value).strip()
    return cleaned or None


def _append_wrap(base: str | None, extra: str) -> str | None:
    extra = _strip_trailing_serial(_title_person(extra))
    if not extra:
        return base
    if base is None:
        return extra
    if extra.lower() in base.lower():
        return base
    return f"{base} {extra}"


def _prefer_person(old: str | None, new: str | None) -> str | None:
    old = _strip_trailing_serial(old)
    if not new:
        return old
    if not old:
        return new
    if len(new) > len(old) and old.lower() in new.lower():
        return new
    return old


def parse_labeled_lines(lines: list[str]) -> dict:
    """Map 4/5/6 body lines to fields. Unlabeled lines continue the previous name."""
    name: str | None = None
    relation_type: str | None = None
    relative_name: str | None = None
    house_no: str | None = None
    age: int | None = None
    gender: str | None = None
    current: str | None = None

    for raw in lines:
        line = raw.strip()
        if _skip_field_line(line):
            continue
        name_m = _NAME_RE.match(line)
        if name_m and not re.search(r"father|mother|husband|wife", line, re.I):
            if name is None:
                name = _title_person(name_m.group(1))
            current = "name"
            continue
        rel_hit = False
        for pattern, label in _RELATION_PATTERNS:
            m = pattern.match(line)
            if m:
                if relation_type is None:
                    relation_type = label
                    value = _title_person(m.group(1))
                    relative_name = value or None
                current = "relative"
                rel_hit = True
                break
        if rel_hit:
            continue
        if _HOUSE_RE.search(line) or _HNO_RE.search(line):
            parsed = parse_house_line(line)
            if parsed:
                if house_no is None:
                    house_no = parsed
                current = "house"
                continue
        next_age, next_gender = parse_age_gender_line(line)
        if next_age is not None or next_gender is not None:
            if age is None:
                age = next_age
            if gender is None:
                gender = next_gender
            current = "age"
            continue
        if current == "name":
            if not _is_wrap_junk(line):
                name = _append_wrap(name, line)
        elif current == "relative":
            if not _is_wrap_junk(line):
                relative_name = _append_wrap(relative_name, line)

    return {
        "name": name,
        "relation_type": relation_type,
        "relative_name": relative_name,
        "house_no": house_no,
        "age": age,
        "gender": gender,
    }


def merge_card_fields(
    record: VoterRecord,
    fields: dict,
    extra_lines: list[str] | None = None,
) -> VoterRecord:
    record.name = _prefer_person(record.name, fields.get("name"))
    if record.relation_type is None and fields.get("relation_type"):
        record.relation_type = fields["relation_type"]
    record.relative_name = _prefer_person(record.relative_name, fields.get("relative_name"))
    if record.house_no is None and fields.get("house_no"):
        record.house_no = fields["house_no"]
    if record.age is None and fields.get("age") is not None:
        record.age = fields["age"]
    if record.gender is None and fields.get("gender"):
        record.gender = fields["gender"]
    if extra_lines:
        seen = set(record.raw_ocr)
        added = [ln for ln in extra_lines if ln and ln not in seen]
        if added:
            record.raw_ocr = list(record.raw_ocr) + added
    return record


def apply_missing_band_text(
    record: VoterRecord,
    *,
    house_text: str | None = None,
    age_text: str | None = None,
    epic_text: str | None = None,
) -> VoterRecord:
    extra: list[str] = []
    if record.epic is None and epic_text:
        extra.append(epic_text)
        found = _normalize_epic(epic_text)
        if found:
            record.epic = found
    if record.house_no is None and house_text:
        extra.append(house_text)
        parsed = parse_house_line(house_text)
        if parsed:
            record.house_no = parsed
    if (record.age is None or record.gender is None) and age_text:
        extra.append(age_text)
        age, gender = parse_age_gender_line(age_text)
        if record.age is None:
            record.age = age
        if record.gender is None:
            record.gender = gender
    if extra:
        record.raw_ocr = list(record.raw_ocr) + extra
    return record


def _normalize_gender(raw: str) -> str:
    value = re.sub(r"[^a-z]", "", raw.strip().lower())
    if value.startswith("female") or value in {
        "femele",
        "femcle",
        "femole",
        "eomolo",
        "femaie",
    }:
        return "Female"
    if value.startswith("male") or value in {"mele", "molo"}:
        return "Male"
    return "Others"


def parse_voter_card(
    tokens: list[OcrToken],
    *,
    page: int,
    meta: PageMeta,
    crop_width: int,
    crop_height: int,
    crop_path: str | None = None,
    mean_confidence: float = 0.0,
) -> VoterRecord:
    lines = group_lines(tokens)
    # Drop photo placeholder.
    lines = [ln for ln in lines if not re.search(r"photo|available", ln, re.I)]

    epic: str | None = None
    serial: int | None = None

    for token in tokens:
        found = _normalize_epic(token.text)
        if found:
            epic = found
            break
    if epic is None:
        blob = " ".join(lines)
        found = _normalize_epic(blob)
        if found:
            epic = found

    # Serial lives in the small top-left box, or as "12 SWD1234567" with the EPIC.
    left_cut = crop_width * 0.28
    top_cut = crop_height * 0.30
    for token in tokens:
        if token.x0 > left_cut or token.y0 > top_cut:
            continue
        if re.search(r"age|gender|name|house", token.text, re.I):
            continue
        if _SERIAL_RE.fullmatch(token.text.strip()):
            serial = int(token.text.strip())
            break
        combo = re.match(r"^(\d{1,4})\s+([A-Z]{3}\d{7})$", token.text.strip(), re.I)
        if combo:
            serial = int(combo.group(1))
            break
    if serial is None:
        for line in lines[:4]:
            combo = re.match(r"^(\d{1,4})\s+[A-Z]{3}\d{7}$", line.strip(), re.I)
            if combo:
                serial = int(combo.group(1))
                break
            m = _SERIAL_RE.fullmatch(line.strip())
            if m:
                serial = int(m.group(1))
                break

    fields = parse_labeled_lines(lines)

    return VoterRecord(
        serial_no=serial,
        epic=epic,
        name=fields["name"],
        relation_type=fields["relation_type"],
        relative_name=fields["relative_name"],
        house_no=fields["house_no"],
        age=fields["age"],
        gender=fields["gender"],
        page=page,
        part_no=meta.part_no,
        constituency=meta.constituency,
        section=meta.section,
        raw_ocr=lines,
        tokens=tokens,
        confidence=mean_confidence,
        crop_path=crop_path,
    )
