from extractors.voter import parse_voter_card
from parser.models import PageMeta
from tests.conftest import CARD_TOKENS, tok


def test_parse_sample_card():
    rec = parse_voter_card(
        CARD_TOKENS,
        page=3,
        meta=PageMeta(part_no=419, constituency="Serilingampally", section="G P R A QTRS, Gachibowli"),
        crop_width=400,
        crop_height=200,
        mean_confidence=0.98,
    )
    assert rec.serial_no == 1
    assert rec.epic == "SWD7756588"
    assert rec.name == "Pavankumar Illa"
    assert rec.relation_type == "Father"
    assert rec.relative_name == "Srinivas Illa"
    assert rec.house_no == "001"
    assert rec.age == 25
    assert rec.gender == "Male"
    assert rec.constituency == "Serilingampally"
    assert rec.section == "G P R A QTRS, Gachibowli"


def test_serial_not_taken_from_age_or_house():
    tokens = [
        tok("Age : 34 Gender : Male", 10, 2),
        tok("SWD7756588", 220, 8),
        tok("Name : Pavankumar Illa", 10, 50),
        tok("Fathers Name: Srinivas Illa", 10, 78),
        tok("House Number : 2-37/8 Dboto", 10, 106),
        tok("1", 8, 10),
    ]
    rec = parse_voter_card(tokens, page=3, meta=PageMeta(), crop_width=400, crop_height=200)
    assert rec.serial_no == 1
    assert rec.age == 34
    assert rec.house_no == "2-37/8 Dboto"


def test_others_relation_and_epic_ocr_noise():
    tokens = [
        tok("5", 8, 8),
        tok("SWD775O588", 220, 8),  # O for 0
        tok("Name : Neelwanti Soni", 10, 50),
        tok("Others: Ram Soni", 10, 78),
        tok("House Number : 1-60/30/8/A/P/1", 10, 106),
        tok("Age : 70 Gender : Female", 10, 134),
    ]
    rec = parse_voter_card(
        tokens, page=3, meta=PageMeta(), crop_width=400, crop_height=200
    )
    assert rec.epic == "SWD7750588"
    assert rec.relation_type == "Other"
    assert rec.gender == "Female"
    assert rec.house_no == "1-60/30/8/A/P/1"


def test_band_parsers_and_fill_does_not_overwrite():
    from extractors.voter import apply_missing_band_text, parse_age_gender_line, parse_house_line

    assert parse_house_line("House Number : 1") == "1"
    assert parse_house_line("House Number : 1-1") == "1-1"
    assert parse_house_line("Houae Number : 1-13. RAYADARGA") == "1-13. RAYADARGA"
    assert parse_house_line("H NO 1-73") == "1-73"
    assert parse_house_line("House Number : 1-37/7/1 Aao :65 Condor :Molo") == "1-37/7/1"
    assert parse_house_line("House Number : H NO 1-87/7/1 A ao : 22 Condor : Molo") == "H NO 1-87/7/1"
    assert parse_age_gender_line("Age : 61 Gender : Female") == (61, "Female")
    assert parse_age_gender_line("Aae : 38 Cender : Mele") == (38, "Male")
    assert parse_age_gender_line("Aao: 38. Gondor : Mele") == (38, "Male")
    assert parse_age_gender_line("Ace:40 Cender • Femele") == (40, "Female")
    assert parse_age_gender_line("A.ae :25Condor: Eomolo") == (25, "Female")
    assert parse_age_gender_line("Aae : 25 Cender : Femcle") == (25, "Female")

    rec = parse_voter_card(
        CARD_TOKENS, page=3, meta=PageMeta(), crop_width=400, crop_height=200
    )
    filled = apply_missing_band_text(
        rec, house_text="House Number : 999", age_text="Age : 99 Gender : Female"
    )
    assert filled.house_no == "001"
    assert filled.age == 25
    assert filled.gender == "Male"

    empty = parse_voter_card(
        [tok("1", 8, 8), tok("SWD7756588", 220, 8), tok("Name : X", 10, 50)],
        page=3,
        meta=PageMeta(),
        crop_width=400,
        crop_height=200,
    )
    empty = apply_missing_band_text(
        empty, house_text="House Number : 1-1", age_text="Age : 27 Gender : Female"
    )
    assert empty.house_no == "1-1"
    assert empty.age == 27
    assert empty.gender == "Female"


def test_wrapped_name_and_relative_six_lines():
    tokens = [
        tok("258", 8, 8),
        tok("SWD7441918", 220, 8),
        tok("Name : MOHAMMED MERAJUDDIN", 10, 50),
        tok("SIDDIQUI", 10, 72),
        tok("Fathers Name: MOHAMMED SHAFIUDDIN", 10, 94),
        tok("SIDDIQUI", 10, 116),
        tok("House Number : 1-37/7/1", 10, 138),
        tok("Age : 65 Gender : Male", 10, 160),
    ]
    rec = parse_voter_card(tokens, page=11, meta=PageMeta(), crop_width=616, crop_height=253)
    assert rec.name == "Mohammed Merajuddin Siddiqui"
    assert rec.relation_type == "Father"
    assert rec.relative_name == "Mohammed Shafiuddin Siddiqui"
    assert rec.house_no == "1-37/7/1"
    assert rec.age == 65
    assert rec.gender == "Male"


def test_name_wrap_five_lines():
    tokens = [
        tok("2", 8, 8),
        tok("SWD6894216", 220, 8),
        tok("Name : MOHAMMED ALIUDDIN", 10, 50),
        tok("SIDDIQUI", 10, 72),
        tok("Fathers Name: Someone", 10, 94),
        tok("House Number : 1-1", 10, 116),
        tok("Age : 40 Gender : Male", 10, 138),
    ]
    rec = parse_voter_card(tokens, page=3, meta=PageMeta(), crop_width=400, crop_height=200)
    assert rec.name == "Mohammed Aliuddin Siddiqui"
    assert rec.relative_name == "Someone"
    assert rec.house_no == "1-1"


def test_relative_wrap_does_not_append_to_voter_name():
    tokens = [
        tok("28", 8, 8),
        tok("SWD6148621", 220, 8),
        tok("Name : Rama Devi Appala", 10, 50),
        tok("Husbands Name: Krishna Yadav Appala", 10, 72),
        tok("Appala", 10, 94),
        tok("1", 10, 100),
        tok("House Number : 1-13. RAYADARGA", 10, 116),
        tok("Age : 49 Gender : Female", 10, 138),
    ]
    rec = parse_voter_card(tokens, page=3, meta=PageMeta(), crop_width=400, crop_height=200)
    assert rec.name == "Rama Devi Appala"
    assert rec.relation_type == "Husband"
    assert rec.relative_name == "Krishna Yadav Appala"
    assert rec.house_no == "1-13. RAYADARGA"


def test_header_parse():
    from extractors.header import parse_page_header

    tokens = [
        tok("Assembly Constituency No and Name : 52-SERILINGAMPALLY"),
        tok("Part No. : 419"),
        tok("Section No and Name 1-G P R A QTRS, Gachibowli"),
        tok("Total Pages 31 - Page 3"),
    ]
    meta = parse_page_header(tokens, fallback_page=3)
    assert meta.constituency_no == 52
    assert meta.constituency == "Serilingampally"
    assert meta.part_no == 419
    assert meta.section == "G P R A QTRS, Gachibowli"
    assert meta.page == 3
    assert meta.total_pages == 31
