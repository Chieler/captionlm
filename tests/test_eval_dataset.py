from captionlm.eval_dataset import build_filing_url, find_press_release_filing, resolve_cik

COMPANY_TICKERS_FIXTURE = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 1018724, "ticker": "AMZN", "title": "Amazon.com Inc."},
}

SUBMISSIONS_FIXTURE = {
    "filings": {
        "recent": {
            "form": ["10-K", "8-K", "4", "8-K"],
            "filingDate": ["2021-01-15", "2021-02-03", "2021-02-04", "2021-06-01"],
            "accessionNumber": ["0001-21-000001", "0001-21-000002", "0001-21-000003", "0001-21-000004"],
            "primaryDocument": ["10k.htm", "pressrelease.htm", "form4.xml", "other.htm"],
        }
    }
}


def test_resolve_cik_found():
    assert resolve_cik("aapl", COMPANY_TICKERS_FIXTURE) == "0000320193"


def test_resolve_cik_not_found():
    assert resolve_cik("nope", COMPANY_TICKERS_FIXTURE) is None


def test_find_press_release_filing_within_window():
    filing = find_press_release_filing("0000320193", "2021-02-01", SUBMISSIONS_FIXTURE, window_days=5)
    assert filing == {"cik": "0000320193", "accession": "0001-21-000002", "primary_document": "pressrelease.htm"}


def test_find_press_release_filing_outside_window():
    filing = find_press_release_filing("0000320193", "2021-05-01", SUBMISSIONS_FIXTURE, window_days=5)
    assert filing is None


def test_build_filing_url():
    url = build_filing_url("0000320193", "0001-21-000002", "pressrelease.htm")
    assert url == "https://www.sec.gov/Archives/edgar/data/320193/000121000002/pressrelease.htm"
