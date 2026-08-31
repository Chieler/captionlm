import json
from pathlib import Path

import pytest

from captionlm.build_eval_set import (
    _normalize_company_name,
    assemble_eval_clip,
    find_exhibit_by_type,
    estimate_call_date,
    extract_call_year,
    find_earnings_release_filing,
    find_exhibit_document,
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
        if "index.json" in url:
            return {"directory": {"item": [{"name": "ex991-pr.htm"}]}}
        return {
            "filings": {
                "recent": {
                    "form": ["8-K"],
                    "filingDate": ["2020-11-10"],
                    "accessionNumber": ["0001-20-000001"],
                    "primaryDocument": ["pr.htm"],
                    "items": ["2.02"],
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


def test_assemble_eval_clip_skips_when_no_item_202_8k(tmp_path, monkeypatch):
    nlp_path = tmp_path / "789.nlp"
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
    wer_tags_path = tmp_path / "789.wer_tag.json"
    wer_tags_path.write_text(json.dumps({"1": {"entity_type": "YEAR"}}), encoding="utf-8")

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
                    "items": ["8.01"],
                }
            }
        }

    monkeypatch.setattr("captionlm.build_eval_set.fetch_json", fake_fetch_json)

    result = assemble_eval_clip(
        "789", "Test Company", 3, "unused.mp3", str(nlp_path), str(wer_tags_path), str(tmp_path / "clips")
    )
    assert result is None


def test_assemble_eval_clip_writes_per_clip_terms_file(tmp_path, monkeypatch):
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
    audio_path.write_bytes(b"\x00")
    out_dir = tmp_path / "clips"

    def fake_fetch_json(url):
        if "company_tickers" in url:
            return {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Test Company"}}
        if "index.json" in url:
            return {"directory": {"item": [{"name": "ex991-pr.htm"}]}}
        return {
            "filings": {
                "recent": {
                    "form": ["8-K"],
                    "filingDate": ["2020-11-10"],
                    "accessionNumber": ["0001-20-000001"],
                    "primaryDocument": ["pr.htm"],
                    "items": ["2.02"],
                }
            }
        }

    monkeypatch.setattr("captionlm.build_eval_set.fetch_json", fake_fetch_json)
    monkeypatch.setattr(
        "captionlm.build_eval_set.fetch_filing_text",
        lambda url: "The kubernetes cluster grew revenue this quarter.",
    )
    monkeypatch.setattr(
        "captionlm.build_eval_set.convert_to_wav",
        lambda mp3, wav: Path(wav).write_bytes(b"RIFF....WAVEfmt "),
    )

    result = assemble_eval_clip(
        "123", "Test Company", 3, str(audio_path), str(nlp_path), str(wer_tags_path), str(out_dir)
    )

    terms_file = out_dir / "123.terms.txt"
    assert terms_file.exists()
    written = [line for line in terms_file.read_text(encoding="utf-8").splitlines() if line]
    assert written == result["terms"]


def _fake_index(monkeypatch, names):
    monkeypatch.setattr(
        "captionlm.build_eval_set.fetch_json",
        lambda url: {"directory": {"item": [{"name": n} for n in names]}},
    )


def test_find_exhibit_document_prefers_ex99_1(monkeypatch):
    _fake_index(monkeypatch, ["ex99-2.htm", "ex99-1.htm"])
    assert find_exhibit_document("0000320193", "0001-20-000001") == "ex99-1.htm"


def test_find_exhibit_document_prefers_dotted_ex99_1(monkeypatch):
    _fake_index(monkeypatch, ["a-ex99d2.htm", "a-ex99.1.htm"])
    assert find_exhibit_document("0000320193", "0001-20-000001") == "a-ex99.1.htm"


def test_find_exhibit_document_falls_back_to_other_ex99(monkeypatch):
    _fake_index(monkeypatch, ["form8k.htm", "ex99-2.htm"])
    assert find_exhibit_document("0000320193", "0001-20-000001") == "ex99-2.htm"


def test_find_exhibit_document_returns_none_without_ex99(monkeypatch):
    _fake_index(monkeypatch, ["form8k.htm", "logo.jpg"])
    assert find_exhibit_document("0000320193", "0001-20-000001") is None


SHARDED_SUBMISSIONS = {
    "filings": {
        "recent": {"form": [], "filingDate": [], "accessionNumber": [], "items": []},
        "files": [
            {
                "name": "CIK0000320193-submissions-001.json",
                "filingFrom": "2020-01-01",
                "filingTo": "2020-12-31",
            },
            {
                "name": "CIK0000320193-submissions-002.json",
                "filingFrom": "2010-01-01",
                "filingTo": "2010-12-31",
            },
        ],
    }
}

SHARD_BODY = {
    "form": ["8-K"],
    "filingDate": ["2020-11-10"],
    "accessionNumber": ["0001-20-000001"],
    "items": ["2.02"],
}


def test_find_earnings_release_filing_searches_shards(monkeypatch):
    monkeypatch.setattr("captionlm.build_eval_set.fetch_json", lambda url: SHARD_BODY)

    filing = find_earnings_release_filing("0000320193", "2020-11-15", SHARDED_SUBMISSIONS)

    assert filing == {"cik": "0000320193", "accession": "0001-20-000001"}


def test_find_earnings_release_filing_skips_out_of_window_shards(monkeypatch):
    fetched = []

    def fake_fetch(url):
        fetched.append(url)
        return SHARD_BODY

    monkeypatch.setattr("captionlm.build_eval_set.fetch_json", fake_fetch)

    find_earnings_release_filing("0000320193", "2020-11-15", SHARDED_SUBMISSIONS)

    assert len(fetched) == 1
    assert "001.json" in fetched[0]


def test_find_earnings_release_filing_without_files_key(monkeypatch):
    def fail_if_called(url):
        raise AssertionError("should not fetch when there are no shards")

    monkeypatch.setattr("captionlm.build_eval_set.fetch_json", fail_if_called)

    submissions = {
        "filings": {
            "recent": {
                "form": ["8-K"],
                "filingDate": ["2020-11-10"],
                "accessionNumber": ["0001-20-000001"],
                "items": ["2.02"],
            }
        }
    }
    filing = find_earnings_release_filing("0000320193", "2020-11-15", submissions)

    assert filing == {"cik": "0000320193", "accession": "0001-20-000001"}


def test_normalize_company_name_strips_suffixes_repeatedly():
    # SEC lists Adtran as "ADTRAN Holdings, Inc." after its 2022 holdco
    # reorg; one pass would leave "ADTRAN HOLDINGS" and never match.
    assert _normalize_company_name("ADTRAN Holdings, Inc.") == "ADTRAN"
    assert _normalize_company_name("Adtran Inc") == "ADTRAN"
    assert _normalize_company_name("Hershey Company") == "HERSHEY"
    assert _normalize_company_name("HERSHEY CO") == "HERSHEY"


def test_resolve_cik_falls_back_to_unique_prefix():
    tickers = {"0": {"cik_str": 72741, "ticker": "ES", "title": "EVERSOURCE ENERGY"}}
    assert resolve_cik_from_company_name("Eversource", tickers) == "0000072741"


def test_resolve_cik_refuses_ambiguous_prefix():
    tickers = {
        "0": {"cik_str": 111111, "ticker": "APLE", "title": "Apple Hospitality REIT"},
        "1": {"cik_str": 222222, "ticker": "AVB", "title": "Apple Valley Bancorp"},
    }
    # No exact match, and "APPLE" prefixes both, so guessing would be wrong
    # half the time. Refusing is the safe answer.
    assert resolve_cik_from_company_name("Apple", tickers) is None


def test_resolve_cik_prefers_exact_match_over_prefix():
    tickers = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc"},
        "1": {"cik_str": 111111, "ticker": "APLE", "title": "Apple Hospitality REIT"},
    }
    # "Apple" normalizes equal to "Apple Inc", so the exact match wins
    # outright and the ambiguous prefix never comes into play.
    assert resolve_cik_from_company_name("Apple", tickers) == "0000320193"


def test_find_exhibit_by_type_reads_escaped_sgml_header(monkeypatch):
    # Cumulus filed its release as cmls20200930earningsre.htm -- no "ex99"
    # substring, so only the SGML <TYPE> tag finds it.
    header = (
        "&lt;TYPE&gt;8-K\n&lt;FILENAME&gt;cmls-20201105.htm\n"
        "&lt;TYPE&gt;EX-99.1\n&lt;FILENAME&gt;cmls20200930earningsre.htm\n"
        "&lt;TYPE&gt;GRAPHIC\n&lt;FILENAME&gt;logo.jpg\n"
    )
    monkeypatch.setattr("captionlm.build_eval_set.fetch_url", lambda url: header.encode())
    assert find_exhibit_by_type("0001058623", "0001058623-20-000055") == "cmls20200930earningsre.htm"


def test_find_exhibit_by_type_returns_none_without_ex99(monkeypatch):
    header = "&lt;TYPE&gt;8-K\n&lt;FILENAME&gt;form8k.htm\n"
    monkeypatch.setattr("captionlm.build_eval_set.fetch_url", lambda url: header.encode())
    assert find_exhibit_by_type("0001058623", "0001058623-20-000055") is None
