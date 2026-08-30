"""Assemble a real eval clip set from Earnings-21 + SEC EDGAR: fetch each
call's matching filing, extract terms from it, and write the
<file_id>.wav/.txt/.doc.txt triples captionlm/eval.py expects.

Neither Earnings-21 nor Earnings-22's own metadata ships an exact call
date (confirmed by inspecting both during planning). The date used to
search SEC EDGAR is estimated from a YEAR entity tag already present in
Earnings-21's own transcript annotations, combined with the metadata
CSV's financial_quarter -- a real signal from the call's own content,
but approximate, which is why find_press_release_filing is called with
a widened window (see captionlm/eval_dataset.py's window_days param).
"""
import ast
import json
import re


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
                return int(parts[token_idx])
            except ValueError:
                continue
    return None


def estimate_call_date(year: int, financial_quarter: int) -> str:
    if financial_quarter == 4:
        return f"{year + 1:04d}-02-15"
    month, day = _QUARTER_ESTIMATE[financial_quarter]
    return f"{year:04d}-{month:02d}-{day:02d}"


def _normalize_company_name(name: str) -> str:
    name = re.sub(r"[.,]", "", name.upper())
    for suffix in (" INC", " CORPORATION", " CORP", " CO", " LTD", " LLC", " PLC"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.strip()


def resolve_cik_from_company_name(company_name: str, company_tickers: dict) -> str | None:
    target = _normalize_company_name(company_name)
    for entry in company_tickers.values():
        if _normalize_company_name(entry.get("title", "")) == target:
            return f"{entry['cik_str']:010d}"
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
