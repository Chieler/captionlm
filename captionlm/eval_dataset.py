"""SEC EDGAR HTTP primitives shared by captionlm/build_eval_set.py.

EDGAR's fair-use policy requires a descriptive User-Agent with a real
contact on every request -- set captionlm.config.SEC_CONTACT before
running fetch_json against the live API.
"""
import json
import time
import urllib.error
import urllib.request
from functools import lru_cache

from captionlm.config import SEC_CONTACT

# EDGAR's fair-use policy caps clients at 10 requests/second and answers
# 503 once you cross it. Assembling a clip costs several round-trips, so a
# tight loop over the metadata CSV trips the limit and the clip is lost to
# a transient error rather than a real miss.
_MIN_REQUEST_INTERVAL = 0.15
_RETRY_STATUSES = (429, 503)
_last_request_time = 0.0

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"


def build_filing_url(cik: str, accession: str, primary_document: str) -> str:
    accession_nodashes = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodashes}/{primary_document}"


def fetch_url(url: str, retries: int = 4) -> bytes:
    """GET a URL with EDGAR's required User-Agent, a client-side rate limit,
    and exponential backoff on the throttling statuses."""
    global _last_request_time
    assert "@" in SEC_CONTACT, "Set a real contact email in config.SEC_CONTACT before hitting SEC EDGAR"
    request = urllib.request.Request(url, headers={"User-Agent": SEC_CONTACT})

    for attempt in range(retries):
        elapsed = time.monotonic() - _last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            if e.code not in _RETRY_STATUSES or attempt == retries - 1:
                raise
            time.sleep(2**attempt)
        finally:
            _last_request_time = time.monotonic()
    raise RuntimeError("unreachable")


@lru_cache
def fetch_json(url: str) -> dict:
    return json.loads(fetch_url(url).decode("utf-8"))
