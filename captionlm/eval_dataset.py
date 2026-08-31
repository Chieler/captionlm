"""SEC EDGAR HTTP primitives shared by captionlm/build_eval_set.py.

EDGAR's fair-use policy requires a descriptive User-Agent with a real
contact on every request -- set captionlm.config.SEC_CONTACT before
running fetch_json against the live API.
"""
import json
import urllib.request
from functools import lru_cache

from captionlm.config import SEC_CONTACT

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"


def build_filing_url(cik: str, accession: str, primary_document: str) -> str:
    accession_nodashes = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodashes}/{primary_document}"


@lru_cache
def fetch_json(url: str) -> dict:
    assert "@" in SEC_CONTACT, "Set a real contact email in config.SEC_CONTACT before hitting SEC EDGAR"
    request = urllib.request.Request(url, headers={"User-Agent": SEC_CONTACT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))
