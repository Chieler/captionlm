"""Assemble doc+audio+transcript eval triples: Earnings-21/22 audio and
transcripts (cloned locally via git-lfs, see the plan's Step 5) paired
with the matching SEC EDGAR earnings press release as the "doc" input.

SEC EDGAR's fair-use policy requires a descriptive User-Agent with a real
contact on every request — set captionlm.config.SEC_CONTACT before
running assemble_triple/fetch_json against the live API.
"""
import json
import urllib.request
from datetime import date
from functools import lru_cache

from captionlm.config import SEC_CONTACT

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"


def resolve_cik(ticker: str, company_tickers: dict) -> str | None:
    ticker = ticker.upper()
    for entry in company_tickers.values():
        if entry.get("ticker", "").upper() == ticker:
            return f"{entry['cik_str']:010d}"
    return None


def find_press_release_filing(
    cik: str, call_date: str, submissions: dict, window_days: int = 5
) -> dict | None:
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    target = date.fromisoformat(call_date)
    best = None
    best_delta = None
    for form, filing_date, accession, primary_doc in zip(forms, filing_dates, accessions, primary_docs):
        if form != "8-K":
            continue
        delta = abs((date.fromisoformat(filing_date) - target).days)
        if delta > window_days:
            continue
        if best_delta is None or delta < best_delta:
            best = {"cik": cik, "accession": accession, "primary_document": primary_doc}
            best_delta = delta
    return best


def build_filing_url(cik: str, accession: str, primary_document: str) -> str:
    accession_nodashes = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodashes}/{primary_document}"


@lru_cache
def fetch_json(url: str) -> dict:
    assert "@" in SEC_CONTACT, "Set a real contact email in config.SEC_CONTACT before hitting SEC EDGAR"
    request = urllib.request.Request(url, headers={"User-Agent": SEC_CONTACT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def assemble_triple(ticker: str, call_date: str) -> dict | None:
    company_tickers = fetch_json(COMPANY_TICKERS_URL)
    cik = resolve_cik(ticker, company_tickers)
    if cik is None:
        return None

    submissions = fetch_json(SUBMISSIONS_URL_TEMPLATE.format(cik=cik))
    filing = find_press_release_filing(cik, call_date, submissions)
    if filing is None:
        return None

    filing["url"] = build_filing_url(filing["cik"], filing["accession"], filing["primary_document"])
    return filing
