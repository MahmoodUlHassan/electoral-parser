from __future__ import annotations

import re

from parser.models import OcrToken, PageMeta

_AC_RE = re.compile(
    r"Assembly\s+Constituency\s+No\s+and\s+Name\s*[:\-]?\s*(\d+)\s*[-–]\s*(.+)",
    re.I,
)
_PART_RE = re.compile(r"Part\s*No\.?\s*[:\-]?\s*(\d+)", re.I)
_SECTION_RE = re.compile(
    r"Section\s+No\s+and\s+Name\s*[:\-]?\s*(\d+)\s*[-–]\s*(.+)",
    re.I,
)
_PAGE_RE = re.compile(r"Total\s+Pages\s+(\d+)\s*[-–]?\s*Page\s+(\d+)", re.I)
_AGE_ON_RE = re.compile(r"Age\s+as\s+on\s+(\d{2}-\d{2}-\d{4})", re.I)
_PUB_RE = re.compile(r"Date\s+of\s+Publication\s*[-:]?\s*(\d{2}-\d{2}-\d{4})", re.I)
_MALE_RE = re.compile(r"\bMale\b", re.I)
_TOTAL_RE = re.compile(r"\bTotal\b", re.I)


def _blob(tokens: list[OcrToken]) -> str:
    return " ".join(t.text for t in tokens)


def _title_place(value: str) -> str:
    value = value.strip(" :|-")
    value = re.sub(r"\s+", " ", value)
    if value.isupper():
        return value.title()
    return value


def parse_page_header(tokens: list[OcrToken], fallback_page: int) -> PageMeta:
    text = _blob(tokens)
    meta = PageMeta(page=fallback_page)
    if m := _AC_RE.search(text):
        meta.constituency_no = int(m.group(1))
        name = m.group(2)
        name = re.split(r"\s+Part\s+No|\s+Section\s+No", name, maxsplit=1)[0]
        meta.constituency = _title_place(name)
    if m := _PART_RE.search(text):
        meta.part_no = int(m.group(1))
    if m := _SECTION_RE.search(text):
        meta.section_no = int(m.group(1))
        rest = m.group(2)
        rest = re.split(
            r"\s+\d+\s+[A-Z]{3}\d{7}|\s+Part\s+No|\s+Age\s+as|\s+Total\s+Pages|\s+Date\s+of",
            rest,
            maxsplit=1,
        )[0]
        meta.section = rest.strip(" :|-")
    if m := _PAGE_RE.search(text):
        meta.total_pages = int(m.group(1))
        meta.page = int(m.group(2))
    if m := _AGE_ON_RE.search(text):
        meta.qualifying_date = m.group(1)
    if m := _PUB_RE.search(text):
        meta.publication_date = m.group(1)
    return meta


def parse_cover_totals(tokens: list[OcrToken]) -> PageMeta:
    """Pull expected elector counts off the cover (page 1) table."""
    meta = parse_page_header(tokens, fallback_page=1)
    numbers = [int(t.text) for t in tokens if t.text.isdigit() and len(t.text) <= 5]
    # Cover summary typically ends with Male, Female, Third Gender, Total.
    # Last four substantial counts are the safest heuristic.
    large = [n for n in numbers if n >= 0]
    if len(large) >= 4:
        meta.expected_male = large[-4]
        meta.expected_female = large[-3]
        meta.expected_others = large[-2]
        meta.expected_total = large[-1]
    return meta


def merge_meta(base: PageMeta, extra: PageMeta) -> PageMeta:
    for field in extra.__dataclass_fields__:
        value = getattr(extra, field)
        if value is not None and getattr(base, field) is None:
            setattr(base, field, value)
    return base
