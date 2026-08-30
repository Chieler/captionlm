import json
import pytest

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


def test_estimate_call_date_invalid_quarter():
    with pytest.raises(ValueError, match="financial_quarter must be 1-4, got 5"):
        estimate_call_date(2020, 5)


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


from pathlib import Path

from captionlm.build_eval_set import assemble_eval_clip


def test_assemble_eval_clip_writes_expected_files(tmp_path, monkeypatch):
    nlp_path = tmp_path / "123.nlp"
    _write_nlp_fixture(
        nlp_path,
        [
            "Good|0||||UC|[]|[]",
            "morning|0||||LC|[]|[]",
            "third|0||||LC|[]|[]",
            "quarter|0||||LC|[]|[]",
            "2020|0||||CA|[]|['1']",
        ],
    )
    wer_tags_path = tmp_path / "123.wer_tag.json"
    wer_tags_path.write_text(json.dumps({"1": {"entity_type": "YEAR"}}), encoding="utf-8")
    audio_path = tmp_path / "123.mp3"
    audio_path.write_bytes(b"\x00")  # never read; convert_to_wav is monkeypatched below
    out_dir = tmp_path / "clips"

    def fake_fetch_json(url):
        if "company_tickers" in url:
            return {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Test Company"}}
        return {
            "filings": {
                "recent": {
                    "form": ["8-K"],
                    "filingDate": ["2020-11-10"],
                    "accessionNumber": ["0001-20-000001"],
                    "primaryDocument": ["pr.htm"],
                }
            }
        }

    monkeypatch.setattr("captionlm.build_eval_set.fetch_json", fake_fetch_json)
    monkeypatch.setattr(
        "captionlm.build_eval_set.fetch_filing_text",
        lambda url: "The kubernetes cluster grew revenue this quarter.",
    )

    written_wavs = []

    def fake_convert(mp3_path, wav_path):
        written_wavs.append(wav_path)
        Path(wav_path).write_bytes(b"RIFF....WAVEfmt ")

    monkeypatch.setattr("captionlm.build_eval_set.convert_to_wav", fake_convert)

    result = assemble_eval_clip(
        "123", "Test Company", 3, str(audio_path), str(nlp_path), str(wer_tags_path), str(out_dir)
    )

    assert result is not None
    assert result["file_id"] == "123"
    assert (out_dir / "123.wav").exists()
    assert (out_dir / "123.txt").read_text(encoding="utf-8") == "Good morning third quarter 2020"
    assert "kubernetes" in (out_dir / "123.doc.txt").read_text(encoding="utf-8").lower()
    assert written_wavs == [str(out_dir / "123.wav")]


def test_assemble_eval_clip_skips_when_no_year_found(tmp_path):
    nlp_path = tmp_path / "456.nlp"
    _write_nlp_fixture(nlp_path, ["Good|0||||UC|[]|[]"])
    wer_tags_path = tmp_path / "456.wer_tag.json"
    wer_tags_path.write_text(json.dumps({}), encoding="utf-8")

    result = assemble_eval_clip(
        "456", "Test Company", 3, "unused.mp3", str(nlp_path), str(wer_tags_path), str(tmp_path / "clips")
    )
    assert result is None
