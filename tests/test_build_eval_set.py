import json

from captionlm.build_eval_set import (
    estimate_call_date,
    extract_call_year,
    reconstruct_transcript_text,
    resolve_cik_from_company_name,
)

COMPANY_TICKERS_FIXTURE = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc"},
    "1": {"cik_str": 1018724, "ticker": "AMZN", "title": "Amazon.com Inc"},
}


def _write_nlp_fixture(path, rows):
    header = "token|speaker|ts|endTs|punctuation|case|tags|wer_tags\n"
    body = "\n".join(rows)
    path.write_text(header + body + "\n", encoding="utf-8")


def test_extract_call_year_finds_year_entity(tmp_path):
    nlp_path = tmp_path / "123.nlp"
    _write_nlp_fixture(
        nlp_path,
        [
            "Good|0||||UC|[]|[]",
            "morning|0||||LC|[]|[]",
            "2020|0||||CA|[]|['1']",
        ],
    )
    wer_tags_path = tmp_path / "123.wer_tag.json"
    wer_tags_path.write_text(json.dumps({"1": {"entity_type": "YEAR"}}), encoding="utf-8")

    assert extract_call_year(str(nlp_path), str(wer_tags_path)) == 2020


def test_extract_call_year_returns_none_when_no_year_tag(tmp_path):
    nlp_path = tmp_path / "123.nlp"
    _write_nlp_fixture(nlp_path, ["Good|0||||UC|[]|[]", "morning|0||||LC|[]|[]"])
    wer_tags_path = tmp_path / "123.wer_tag.json"
    wer_tags_path.write_text(json.dumps({}), encoding="utf-8")

    assert extract_call_year(str(nlp_path), str(wer_tags_path)) is None


def test_estimate_call_date_quarters():
    assert estimate_call_date(2020, 1) == "2020-05-15"
    assert estimate_call_date(2020, 2) == "2020-08-15"
    assert estimate_call_date(2020, 3) == "2020-11-15"
    assert estimate_call_date(2020, 4) == "2021-02-15"


def test_resolve_cik_from_company_name_found():
    assert resolve_cik_from_company_name("Apple Inc", COMPANY_TICKERS_FIXTURE) == "0000320193"


def test_resolve_cik_from_company_name_not_found():
    assert resolve_cik_from_company_name("Nonexistent Corp", COMPANY_TICKERS_FIXTURE) is None


def test_reconstruct_transcript_text(tmp_path):
    nlp_path = tmp_path / "123.nlp"
    _write_nlp_fixture(
        nlp_path,
        [
            "Good|0||||UC|[]|[]",
            "morning|0||||LC|[]|[]",
            "everyone|0|||.|LC|[]|[]",
        ],
    )
    assert reconstruct_transcript_text(str(nlp_path)) == "Good morning everyone."
