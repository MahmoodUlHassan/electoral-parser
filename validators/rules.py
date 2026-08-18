from __future__ import annotations

import re

from parser.config import DEFAULT_LAYOUT, LayoutProfile
from parser.models import VoterRecord

_EPIC_RE = re.compile(DEFAULT_LAYOUT.epic_pattern)
_ALLOWED_GENDER = {"Male", "Female", "Others"}
_ALLOWED_RELATION = {"Father", "Mother", "Husband", "Wife", "Other"}


def validate_voter(record: VoterRecord, layout: LayoutProfile = DEFAULT_LAYOUT) -> VoterRecord:
    errors: list[str] = []
    if record.epic is None:
        errors.append("missing_epic")
    elif not _EPIC_RE.fullmatch(record.epic):
        errors.append("invalid_epic")
    if record.age is None:
        errors.append("missing_age")
    elif not (layout.age_min <= record.age <= layout.age_max):
        errors.append("invalid_age")
    if record.gender is None:
        errors.append("missing_gender")
    elif record.gender not in _ALLOWED_GENDER:
        errors.append("invalid_gender")
    if record.name is None:
        errors.append("missing_name")
    if record.serial_no is None:
        errors.append("missing_serial")
    if record.relation_type is not None and record.relation_type not in _ALLOWED_RELATION:
        errors.append("invalid_relation")
    record.errors = errors
    record.valid = len(errors) == 0
    return record
