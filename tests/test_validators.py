from extractors.voter import parse_voter_card
from parser.models import PageMeta
from tests.conftest import CARD_TOKENS
from validators.rules import validate_voter


def test_valid_sample():
    rec = parse_voter_card(
        CARD_TOKENS, page=3, meta=PageMeta(), crop_width=400, crop_height=200
    )
    rec = validate_voter(rec)
    assert rec.valid
    assert rec.errors == []


def test_age_and_epic_rules():
    from parser.models import VoterRecord

    rec = VoterRecord(
        serial_no=1,
        epic="SWD77565",
        name="Ada",
        relation_type="Father",
        relative_name="Bob",
        house_no="1",
        age=12,
        gender="Unknown",
        page=3,
        part_no=1,
        constituency="X",
        section="Y",
    )
    rec = validate_voter(rec)
    assert not rec.valid
    assert "invalid_epic" in rec.errors
    assert "invalid_age" in rec.errors
    assert "invalid_gender" in rec.errors
