"""Assemble a real eval clip set from Earnings-21 + SEC EDGAR: fetch each
call's matching filing, extract terms from it, and write the
<file_id>.wav/.txt/.doc.txt triples captionlm/eval.py expects.

Neither Earnings-21 nor Earnings-22's own metadata ships an exact call
date (confirmed by inspecting both during planning). The date used to
search SEC EDGAR is estimated from a YEAR entity tag already present in
Earnings-21's own transcript annotations, combined with the metadata
CSV's financial_quarter -- a real signal from the call's own content,
but approximate, which is why find_earnings_release_filing is called
with a widened window_days.
"""
import argparse
import ast
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from datetime import date, timedelta

from captionlm.config import SEC_CONTACT
from captionlm.doc_import import extract_text
from captionlm.eval_dataset import (
    COMPANY_TICKERS_URL,
    SUBMISSIONS_URL_TEMPLATE,
    build_filing_url,
    fetch_json,
    fetch_url,
)
from captionlm.term_extraction import extract_terms, write_term_list

_QUARTER_ESTIMATE = {1: (5, 15), 2: (8, 15), 3: (11, 15)}


def extract_call_year(nlp_path: str, wer_tags_path: str) -> int | None:
    with open(wer_tags_path, encoding="utf-8") as f:
        wer_tags = json.load(f)
    year_ids = {tag_id for tag_id, info in wer_tags.items() if info.get("entity_type") == "YEAR"}
    if not year_ids:
        return None

    with open(nlp_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    header = lines[0].split("|")
    token_idx = header.index("token")
    wer_tags_idx = header.index("wer_tags")

    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) <= wer_tags_idx:
            continue
        ids = ast.literal_eval(parts[wer_tags_idx])
        if any(tag_id in year_ids for tag_id in ids):
            try:
                year = int(parts[token_idx])
            except ValueError:
                continue
            if 1990 <= year <= 2100:
                return year
    return None


def estimate_call_date(year: int, financial_quarter: int) -> str:
    if financial_quarter == 4:
        return f"{year + 1:04d}-02-15"
    if financial_quarter not in _QUARTER_ESTIMATE:
        raise ValueError(f"financial_quarter must be 1-4, got {financial_quarter}")
    month, day = _QUARTER_ESTIMATE[financial_quarter]
    return f"{year:04d}-{month:02d}-{day:02d}"


_NAME_SUFFIXES = (
    " INCORPORATED", " INC", " CORPORATION", " CORP", " COMPANY", " CO",
    " LIMITED", " LTD", " LLC", " PLC", " HOLDINGS", " HOLDING", " GROUP",
    " THE", " & CO",
)


def _normalize_company_name(name: str) -> str:
    """Upper-case, drop punctuation, and strip corporate suffixes REPEATEDLY.

    One pass is not enough: SEC lists Adtran as "ADTRAN Holdings, Inc.",
    so stripping only " INC" leaves "ADTRAN HOLDINGS", which never matches
    Earnings-21's "Adtran Inc".
    """
    name = re.sub(r"[.,]", "", name.upper())
    changed = True
    while changed:
        changed = False
        for suffix in _NAME_SUFFIXES:
            if name.endswith(suffix) and len(name) > len(suffix):
                name = name[: -len(suffix)].strip()
                changed = True
                break
    return name.strip()


def resolve_cik_from_company_name(company_name: str, company_tickers: dict) -> str | None:
    """Resolve a CIK by company name, exact normalized match first.

    Falls back to a UNIQUE prefix match, because Earnings-21 names are
    often shorter than SEC's ("Eversource" vs "EVERSOURCE ENERGY"). The
    uniqueness requirement is what keeps the fallback safe: "APPLE"
    prefixes both "APPLE INC" and "APPLE HOSPITALITY REIT", so it is
    ambiguous and correctly refused rather than guessed.
    """
    target = _normalize_company_name(company_name)
    if not target:
        return None

    prefix_hits = []
    for entry in company_tickers.values():
        title = _normalize_company_name(entry.get("title", ""))
        if title == target:
            return f"{entry['cik_str']:010d}"
        if title.startswith(target + " ") or target.startswith(title + " "):
            prefix_hits.append(entry)

    if len(prefix_hits) == 1:
        return f"{prefix_hits[0]['cik_str']:010d}"
    return None


def reconstruct_transcript_text(nlp_path: str) -> str:
    with open(nlp_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    header = lines[0].split("|")
    token_idx = header.index("token")
    punct_idx = header.index("punctuation")

    words = []
    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) <= punct_idx:
            continue
        words.append(parts[token_idx] + parts[punct_idx])
    return " ".join(words)


def iter_filing_blocks(submissions: dict, call_date: str, window_days: int):
    """Yield filings.recent, then each filings.files[] shard whose date
    range overlaps the search window.

    submissions.json caps filings.recent at ~1000 entries; for a
    high-volume filer that is well under the two years Earnings-21
    covers. Shard bodies carry the column arrays at the JSON root.
    """
    filings = submissions.get("filings", {})
    yield filings.get("recent", {})

    target = date.fromisoformat(call_date)
    window_start = target - timedelta(days=window_days)
    window_end = target + timedelta(days=window_days)
    for shard in filings.get("files", []):
        shard_from = date.fromisoformat(shard["filingFrom"])
        shard_to = date.fromisoformat(shard["filingTo"])
        if shard_to < window_start or shard_from > window_end:
            continue
        yield fetch_json(f"https://data.sec.gov/submissions/{shard['name']}")


def find_earnings_release_filing(
    cik: str, call_date: str, submissions: dict, window_days: int = 60
) -> dict | None:
    target = date.fromisoformat(call_date)
    best = None
    best_delta = None
    for block in iter_filing_blocks(submissions, call_date, window_days):
        forms = block.get("form", [])
        filing_dates = block.get("filingDate", [])
        accessions = block.get("accessionNumber", [])
        items = block.get("items", [])
        for form, filing_date, accession, item_str in zip(forms, filing_dates, accessions, items):
            if form != "8-K":
                continue
            if "2.02" not in (item_str or "").split(","):
                continue
            delta = abs((date.fromisoformat(filing_date) - target).days)
            if delta > window_days:
                continue
            if best_delta is None or delta < best_delta:
                best = {"cik": cik, "accession": accession}
                best_delta = delta
    return best


def _exhibit_rank(name: str) -> int:
    """0 for EX-99.1 (the press release), 1 for any other EX-99.x.

    EX-99.2 on an earnings 8-K is usually the slide deck or supplemental
    tables; term extraction over tabular text yields column headers that
    are never spoken on the call.
    """
    lowered = name.lower()
    if any(marker in lowered for marker in ("ex99-1", "ex99.1", "ex991", "ex99_1", "ex99d1", "ex-99.1")):
        return 0
    return 1


def find_exhibit_by_type(cik: str, accession: str) -> str | None:
    """Resolve the EX-99.1 document from the filing's SGML header.

    Filers name exhibit files freely -- Cumulus Media filed its Q3 2020
    release as `cmls20200930earningsre.htm`, which contains no "ex99"
    substring at all, so matching on filename silently loses those
    filings. The <TYPE>/<FILENAME> pairs in the index-headers document are
    EDGAR's own authoritative answer, so ask it instead of guessing.
    """
    accession_nodashes = accession.replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodashes}/"
        f"{accession}-index-headers.html"
    )
    try:
        html = fetch_url(url).decode("utf-8", "replace")
    except Exception:
        return None

    # The header is HTML-escaped SGML: &lt;TYPE&gt;EX-99.1 ... &lt;FILENAME&gt;foo.htm
    pairs = re.findall(
        r"(?:&lt;|<)TYPE(?:&gt;|>)\s*([^\s<&]+).*?(?:&lt;|<)FILENAME(?:&gt;|>)\s*([^\s<&]+)",
        html,
        re.S,
    )
    exhibits = [(t.upper(), f) for t, f in pairs if t.upper().startswith("EX-99")]
    if not exhibits:
        return None
    exact = [f for t, f in exhibits if t == "EX-99.1"]
    return exact[0] if exact else exhibits[0][1]


def find_exhibit_document(cik: str, accession: str) -> str | None:
    """Locate the earnings press release exhibit in an 8-K.

    Asks EDGAR's SGML header for the document typed EX-99.1 first, and
    only falls back to the filename heuristic when the header is
    unavailable or lists no EX-99 document.
    """
    by_type = find_exhibit_by_type(cik, accession)
    if by_type:
        return by_type

    accession_nodashes = accession.replace("-", "")
    index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodashes}/index.json"
    index = fetch_json(index_url)
    candidates = [
        item.get("name", "")
        for item in index.get("directory", {}).get("item", [])
        if "ex99" in item.get("name", "").lower() or "ex-99" in item.get("name", "").lower()
    ]
    if not candidates:
        return None
    # min() is stable, so equal-ranked candidates keep index.json order.
    return min(candidates, key=_exhibit_rank)


def fetch_filing_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": SEC_CONTACT})
    fd, tmp_path = tempfile.mkstemp(suffix=".htm")
    try:
        with os.fdopen(fd, "wb") as tmp_f, urllib.request.urlopen(request, timeout=30) as response:
            tmp_f.write(response.read())
        return extract_text(tmp_path, max_length=200_000)
    finally:
        os.remove(tmp_path)


def convert_to_wav(mp3_path: str, wav_path: str) -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode(errors='replace')[-2000:]}") from e


def assemble_eval_clip(
    file_id: str,
    company_name: str,
    financial_quarter: int,
    audio_path: str,
    nlp_path: str,
    wer_tags_path: str,
    out_dir: str,
) -> dict | None:
    year = extract_call_year(nlp_path, wer_tags_path)
    if year is None:
        print(f"skip {file_id}: no YEAR entity found in transcript")
        return None

    company_tickers = fetch_json(COMPANY_TICKERS_URL)
    cik = resolve_cik_from_company_name(company_name, company_tickers)
    if cik is None:
        print(f"skip {file_id}: no CIK match for company '{company_name}'")
        return None

    call_date = estimate_call_date(year, financial_quarter)
    submissions = fetch_json(SUBMISSIONS_URL_TEMPLATE.format(cik=cik))
    filing = find_earnings_release_filing(cik, call_date, submissions, window_days=60)
    if filing is None:
        print(f"skip {file_id}: no Item 2.02 8-K found near {call_date} for CIK {cik}")
        return None

    exhibit_name = find_exhibit_document(filing["cik"], filing["accession"])
    if exhibit_name is None:
        print(f"skip {file_id}: no EX-99.1 exhibit found in 8-K {filing['accession']} for CIK {cik}")
        return None

    filing_url = build_filing_url(filing["cik"], filing["accession"], exhibit_name)
    filing_text = fetch_filing_text(filing_url)
    terms = extract_terms(filing_text)

    os.makedirs(out_dir, exist_ok=True)
    convert_to_wav(audio_path, os.path.join(out_dir, f"{file_id}.wav"))
    with open(os.path.join(out_dir, f"{file_id}.txt"), "w", encoding="utf-8") as f:
        f.write(reconstruct_transcript_text(nlp_path))
    with open(os.path.join(out_dir, f"{file_id}.doc.txt"), "w", encoding="utf-8") as f:
        f.write(filing_text)
    write_term_list(terms, os.path.join(out_dir, f"{file_id}.terms.txt"))

    return {"file_id": file_id, "terms": terms, "filing_url": filing_url}


def main():
    parser = argparse.ArgumentParser(
        description="Assemble an eval clip set from a cloned Earnings-21 directory + live SEC EDGAR"
    )
    parser.add_argument("earnings21_dir", help="Path to the cloned earnings21/ directory")
    parser.add_argument("--out", default="eval_data/clips", help="Output clip directory")
    parser.add_argument("--terms-out", default="eval_data/terms.txt", help="Combined term list output path")
    parser.add_argument("--limit", type=int, default=5, help="Max number of clips to successfully assemble")
    parser.add_argument(
        "--max-attempts", type=int, default=20, help="Max number of rows to process (successes + skips)"
    )
    args = parser.parse_args()

    if os.path.exists(args.out):
        out_abs = os.path.abspath(args.out)
        dangerous = {os.path.abspath(os.sep), os.path.abspath(os.path.expanduser("~"))}
        if out_abs in dangerous or len(out_abs.rstrip(os.sep).split(os.sep)) <= 3:
            raise SystemExit(f"refusing to rmtree suspicious --out path: {out_abs}")
        print(f"Clearing stale output directory {args.out}")
        shutil.rmtree(args.out)

    metadata_path = os.path.join(args.earnings21_dir, "earnings21-file-metadata.csv")
    all_terms: set[str] = set()
    assembled = 0
    attempted = 0

    with open(metadata_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if assembled >= args.limit or attempted >= args.max_attempts:
                break
            file_id = row["file_id"]
            audio_path = os.path.join(args.earnings21_dir, "media", f"{file_id}.mp3")
            nlp_path = os.path.join(args.earnings21_dir, "transcripts", "nlp_references", f"{file_id}.nlp")
            wer_tags_path = os.path.join(
                args.earnings21_dir, "transcripts", "wer_tags", f"{file_id}.wer_tag.json"
            )
            if not (os.path.exists(audio_path) and os.path.exists(nlp_path) and os.path.exists(wer_tags_path)):
                continue

            attempted += 1
            try:
                result = assemble_eval_clip(
                    file_id,
                    row["company_name"],
                    int(row["financial_quarter"]),
                    audio_path,
                    nlp_path,
                    wer_tags_path,
                    args.out,
                )
            except Exception as e:
                print(f"skip {file_id}: {type(e).__name__}: {e}")
                continue
            if result is None:
                continue
            all_terms.update(result["terms"])
            assembled += 1

    if assembled > 0:
        os.makedirs(os.path.dirname(args.terms_out) or ".", exist_ok=True)
        write_term_list(sorted(all_terms), args.terms_out)
        print(f"Assembled {assembled}/{attempted} clips into {args.out}; terms written to {args.terms_out}")
    else:
        print(f"Assembled {assembled}/{attempted} clips into {args.out}; nothing assembled, skipping terms.txt write")


if __name__ == "__main__":
    main()
