# Glue and Eval Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fold `file_import.py` into the `captionlm` package, build the
missing orchestration that turns a real Earnings-21 call + its SEC filing
into an eval clip, and run `eval.py` against real data for real WER/F-score
numbers.

**Architecture:** `captionlm/doc_import.py` (moved + refactored from
`file_import.py`) extracts plain text from any local file. A new
`captionlm/build_eval_set.py` chains it with the already-built
`term_extraction.py` and `eval_dataset.py` to assemble
`<file_id>.wav`/`.txt`/`.doc.txt` triples from Earnings-21 + a live SEC
EDGAR lookup, using entity tags already present in Earnings-21's own
transcripts to estimate each call's date (neither dataset ships one
directly). `eval.py` (already built, unchanged) then scores the result.

**Tech Stack:** Python 3.12, `extractous` (unchanged dependency), stdlib
`urllib`/`csv`/`ast`/`subprocess`, `ffmpeg` (external tool, already
installed), git-lfs (external tool, local-scope install only).

**Spec:** `docs/superpowers/specs/2026-08-30-glue-and-eval-pipeline-design.md`

## Global Constraints

- Every commit ends with a `Co-Authored-By: Claude Sonnet 5
  <noreply@anthropic.com>` trailer, matching this repo's other commits.
- Test runner: `pytest`, run from the repo root (`venv/bin/pytest`).
- `captionlm/eval_dataset.py` is NOT modified by this plan — its
  functions (`fetch_json`, `find_press_release_filing`, `build_filing_url`,
  `COMPANY_TICKERS_URL`, `SUBMISSIONS_URL_TEMPLATE`) are imported and used
  as-is. `resolve_cik` (ticker-based) is not used here — Earnings-21 gives
  a company name, not a ticker; `resolve_cik_from_company_name` (new, in
  `build_eval_set.py`) matches against `company_tickers.json`'s `"title"`
  field instead.
- `find_press_release_filing` is called with `window_days=60`, not its
  default of 5 — the call date used here is an estimate (see Task 2), not
  an exact date.
- git-lfs: `git lfs install` writes filter config into whatever git repo
  it's run in. Only ever run it with `--local` scope, inside the
  `vendor_data/` clone created in Task 4 — never `--global`, never without
  `--local`, never in this repo's own `.git`.
- SEC EDGAR requires a real contact in `captionlm.config.SEC_CONTACT`
  before any live request — Task 4 sets it to `chielerli@gmail.com`
  (confirmed with the user).
- `eval_data/` (new, generated) and `vendor_data/` (already gitignored by
  a prior plan) hold generated/cloned data — never commit their contents.

---

### Task 1: `doc_import.py` — move and refactor `file_import.py`

**Files:**
- Create: `captionlm/doc_import.py`
- Delete: `file_import.py`
- Test: `tests/test_doc_import.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (this is the first task).
- Produces: `captionlm.doc_import.extract_text(path: str, max_length: int
  = 50_000) -> str`. Used by `captionlm/build_eval_set.py` (Tasks 2-3).

**Context:** The current `file_import.py` (repo root) is a script with
module-level side effects — it creates an `Extractor`, lists and renames
every file in a hardcoded `static_assets/` path, and prints on import.
Nothing in the repo currently imports it (`grep -rn "import file_import"`
finds no hits). This task turns its extraction logic into a real,
side-effect-free function.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_doc_import.py
from pathlib import Path

from captionlm.doc_import import extract_text

FIXTURE = (
    Path(__file__).parent.parent
    / "static_assets"
    / "Philo_Homes_Organization_Structure_Proposal.docx"
)


def test_extract_text_handles_google_docs_tika_198():
    assert FIXTURE.exists(), f"missing fixture file: {FIXTURE}"
    text = extract_text(str(FIXTURE))
    assert len(text) > 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_doc_import.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'captionlm.doc_import'`

- [ ] **Step 3: Delete `file_import.py`, create `captionlm/doc_import.py`**

```bash
git rm file_import.py
```

```python
# captionlm/doc_import.py
"""Extract plain text from local files (docx, pdf, csv, and whatever
else extractous/Tika support), including a workaround for
Google-Docs-exported .docx files that trigger Tika's TIKA-198 error."""
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from extractous import Extractor

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W_NS)
W = f"{{{W_NS}}}"


def unwrap_google_docs_sdt(xml_bytes):
    """Unwrap <w:sdt> tags Google Docs inserts (tag="goog_rdk_*") around runs.

    Apache POI/Tika's OOXMLParser throws TIKA-198 on these when a .docx was
    exported from Google Docs. They carry no real content control, so
    replacing each with its own sdtContent children is safe.
    """
    root = ET.fromstring(xml_bytes)
    parent_map = {child: parent for parent in root.iter() for child in parent}
    for sdt in list(root.iter(W + "sdt")):
        sdt_pr = sdt.find(W + "sdtPr")
        tag_el = sdt_pr.find(W + "tag") if sdt_pr is not None else None
        tag_val = tag_el.get(W + "val") if tag_el is not None else ""
        if not (tag_val or "").startswith("goog_rdk"):
            continue
        content = sdt.find(W + "sdtContent")
        parent = parent_map[sdt]
        idx = list(parent).index(sdt)
        parent.remove(sdt)
        for i, child in enumerate(content if content is not None else []):
            parent.insert(idx + i, child)
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True)


def fix_google_docs_docx(path):
    """Return a temp copy of a Google-Docs-exported .docx with the
    goog_rdk sdt wrappers stripped, so Tika can parse it."""
    with zipfile.ZipFile(path) as src:
        fd, tmp_path = tempfile.mkstemp(suffix=".docx")
        with os.fdopen(fd, "wb") as tmp_f, zipfile.ZipFile(tmp_f, "w", zipfile.ZIP_DEFLATED) as dst:
            for item in src.infolist():
                data = src.read(item.filename)
                if item.filename == "word/document.xml":
                    data = unwrap_google_docs_sdt(data)
                dst.writestr(item, data)
    return tmp_path


def extract_text(path: str, max_length: int = 50_000) -> str:
    extractor = Extractor().set_extract_string_max_length(max_length)
    try:
        result, _metadata = extractor.extract_file_to_string(path)
    except TypeError as e:
        if path.lower().endswith(".docx") and "TIKA-198" in str(e):
            fixed = fix_google_docs_docx(path)
            try:
                result, _metadata = extractor.extract_file_to_string(fixed)
            finally:
                os.remove(fixed)
        else:
            raise
    return result


if __name__ == "__main__":
    static_assets_dir = os.path.join(os.path.dirname(__file__), "..", "static_assets")
    for asset in os.listdir(static_assets_dir):
        p = os.path.join(static_assets_dir, asset)
        print(p, extract_text(p))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_doc_import.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add captionlm/doc_import.py tests/test_doc_import.py
git commit -m "$(cat <<'EOF'
Fold file_import.py into captionlm/ as a real function

extract_text(path) has no module-level side effects (the old script
created an Extractor and renamed/scanned static_assets/ on import).
The TIKA-198 Google-Docs-export workaround is unchanged. max_length
default raised from the old hardcoded 3000 to 50,000 -- that limit was
sized for short reference docs; SEC filings and other real documents
need more headroom.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `build_eval_set.py` — pure functions

**Files:**
- Create: `captionlm/build_eval_set.py`
- Test: `tests/test_build_eval_set.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `captionlm.build_eval_set.extract_call_year(nlp_path: str,
  wer_tags_path: str) -> int | None`,
  `captionlm.build_eval_set.estimate_call_date(year: int,
  financial_quarter: int) -> str` (returns `"YYYY-MM-DD"`),
  `captionlm.build_eval_set.resolve_cik_from_company_name(company_name:
  str, company_tickers: dict) -> str | None`,
  `captionlm.build_eval_set.reconstruct_transcript_text(nlp_path: str) ->
  str`. All four are used by Task 3's `assemble_eval_clip`.

**Context:** Earnings-21's own metadata (`earnings21-file-metadata.csv`:
`file_id, audio_length, sample_rate, company_name, financial_quarter,
sector, speaker_switches, unique_speakers, curator_id`) has no ticker and
no call date. The only real date signal is inside each call's own
transcript: `.nlp` files are pipe-separated token tables
(`token|speaker|ts|endTs|punctuation|case|tags|wer_tags`) where the
`wer_tags` column is a Python-list-literal string of entity IDs (e.g.
`"['4']"` or `"[]"`), resolved via a sidecar `<file_id>.wer_tag.json`
mapping ID → `{"entity_type": "..."}`. One of those entity types is
`"YEAR"` — confirmed on a real file (`4320211`): the token `"2020"` is
tagged `YEAR`. This step's four functions are all pure/local — no
network, no external tools — testable with small synthetic fixtures.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_build_eval_set.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_build_eval_set.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'captionlm.build_eval_set'`

- [ ] **Step 3: Write `captionlm/build_eval_set.py` (pure functions only)**

```python
# captionlm/build_eval_set.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_build_eval_set.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add captionlm/build_eval_set.py tests/test_build_eval_set.py
git commit -m "$(cat <<'EOF'
Add pure date/name-resolution functions for eval set assembly

Neither Earnings-21 nor Earnings-22 ships an exact call date in their
own metadata (verified during planning). extract_call_year pulls a
YEAR entity tag out of Earnings-21's own transcript annotations
instead -- a real signal from the call's own content, combined with
financial_quarter into an approximate date via estimate_call_date.
resolve_cik_from_company_name matches on company name since
Earnings-21 gives no ticker symbol.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `build_eval_set.py` — orchestration

**Files:**
- Modify: `captionlm/build_eval_set.py` (add to the file from Task 2)
- Test: `tests/test_build_eval_set.py` (add to the file from Task 2)

**Interfaces:**
- Consumes: `captionlm.doc_import.extract_text` (Task 1);
  `captionlm.build_eval_set.extract_call_year`,
  `.estimate_call_date`, `.resolve_cik_from_company_name`,
  `.reconstruct_transcript_text` (Task 2, same file);
  `captionlm.eval_dataset.fetch_json`,
  `.find_press_release_filing`, `.build_filing_url`,
  `.COMPANY_TICKERS_URL`, `.SUBMISSIONS_URL_TEMPLATE` (already built,
  unmodified); `captionlm.term_extraction.extract_terms`,
  `.write_term_list` (already built, unmodified);
  `captionlm.config.SEC_CONTACT` (Task 4 sets its value, but this task's
  code already reads it via `doc_import`'s network call).
- Produces: `captionlm.build_eval_set.fetch_filing_text(url: str) -> str`,
  `captionlm.build_eval_set.convert_to_wav(mp3_path: str, wav_path: str)
  -> None`, `captionlm.build_eval_set.assemble_eval_clip(file_id: str,
  company_name: str, financial_quarter: int, audio_path: str, nlp_path:
  str, wer_tags_path: str, out_dir: str) -> dict | None` (returns
  `{"file_id": str, "terms": list[str], "filing_url": str}` or `None` if
  any resolution step fails), `captionlm.build_eval_set.main()`.

**Context:** `find_press_release_filing` is called directly (not via
`eval_dataset.assemble_triple`, which hardcodes `window_days=5`) with
`window_days=60`, since `estimate_call_date` only gives an approximate
target. `convert_to_wav` exists because `eval.py.load_clips` globs
`*.wav`; `parakeet-mlx`'s own audio loader shells out to `ffmpeg -i
<path>` directly and doesn't care about file extension or input sample
rate, so this conversion isn't for the model — it's so `eval_data/clips/`
doesn't contain files whose extension lies about their format.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_build_eval_set.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_build_eval_set.py -v`
Expected: FAIL — `assemble_eval_clip`, `fetch_filing_text`, `convert_to_wav`
not defined yet (`ImportError`/`AttributeError`)

- [ ] **Step 3: Add orchestration functions to `captionlm/build_eval_set.py`**

Add these imports to the top of the file (alongside the existing `ast`,
`json`, `re`):

```python
import argparse
import csv
import os
import subprocess
import tempfile
import urllib.request

from captionlm.config import SEC_CONTACT
from captionlm.doc_import import extract_text
from captionlm.eval_dataset import (
    COMPANY_TICKERS_URL,
    SUBMISSIONS_URL_TEMPLATE,
    build_filing_url,
    fetch_json,
    find_press_release_filing,
)
from captionlm.term_extraction import extract_terms, write_term_list
```

Append these functions after `reconstruct_transcript_text`:

```python
def fetch_filing_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": SEC_CONTACT})
    fd, tmp_path = tempfile.mkstemp(suffix=".htm")
    try:
        with os.fdopen(fd, "wb") as tmp_f, urllib.request.urlopen(request, timeout=30) as response:
            tmp_f.write(response.read())
        return extract_text(tmp_path)
    finally:
        os.remove(tmp_path)


def convert_to_wav(mp3_path: str, wav_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", wav_path],
        check=True,
        capture_output=True,
    )


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
    filing = find_press_release_filing(cik, call_date, submissions, window_days=60)
    if filing is None:
        print(f"skip {file_id}: no 8-K found near {call_date} for CIK {cik}")
        return None

    filing_url = build_filing_url(filing["cik"], filing["accession"], filing["primary_document"])
    filing_text = fetch_filing_text(filing_url)
    terms = extract_terms(filing_text)

    os.makedirs(out_dir, exist_ok=True)
    convert_to_wav(audio_path, os.path.join(out_dir, f"{file_id}.wav"))
    with open(os.path.join(out_dir, f"{file_id}.txt"), "w", encoding="utf-8") as f:
        f.write(reconstruct_transcript_text(nlp_path))
    with open(os.path.join(out_dir, f"{file_id}.doc.txt"), "w", encoding="utf-8") as f:
        f.write(filing_text)

    return {"file_id": file_id, "terms": terms, "filing_url": filing_url}


def main():
    parser = argparse.ArgumentParser(
        description="Assemble an eval clip set from a cloned Earnings-21 directory + live SEC EDGAR"
    )
    parser.add_argument("earnings21_dir", help="Path to the cloned earnings21/ directory")
    parser.add_argument("--out", default="eval_data/clips", help="Output clip directory")
    parser.add_argument("--terms-out", default="eval_data/terms.txt", help="Combined term list output path")
    parser.add_argument("--limit", type=int, default=5, help="Max number of calls to process")
    args = parser.parse_args()

    metadata_path = os.path.join(args.earnings21_dir, "earnings21-file-metadata.csv")
    all_terms: set[str] = set()
    assembled = 0

    with open(metadata_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if assembled >= args.limit:
                break
            file_id = row["file_id"]
            audio_path = os.path.join(args.earnings21_dir, "media", f"{file_id}.mp3")
            nlp_path = os.path.join(args.earnings21_dir, "transcripts", "nlp_references", f"{file_id}.nlp")
            wer_tags_path = os.path.join(
                args.earnings21_dir, "transcripts", "wer_tags", f"{file_id}.wer_tag.json"
            )
            if not (os.path.exists(audio_path) and os.path.exists(nlp_path) and os.path.exists(wer_tags_path)):
                continue

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
                print(f"skip {file_id}: {e}")
                continue
            if result is None:
                continue
            all_terms.update(result["terms"])
            assembled += 1

    os.makedirs(os.path.dirname(args.terms_out) or ".", exist_ok=True)
    write_term_list(sorted(all_terms), args.terms_out)
    print(f"Assembled {assembled} clips into {args.out}; terms written to {args.terms_out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_build_eval_set.py -v`
Expected: PASS (8 tests total: 6 from Task 2 + 2 new)

- [ ] **Step 5: Commit**

```bash
git add captionlm/build_eval_set.py tests/test_build_eval_set.py
git commit -m "$(cat <<'EOF'
Add build_eval_set.py orchestration: filing fetch, clip assembly, CLI

find_press_release_filing is called directly with window_days=60
(not via eval_dataset.assemble_triple, which hardcodes 5) since the
estimated call date is approximate. convert_to_wav exists because
eval.py's load_clips globs *.wav, not because the model needs it --
parakeet-mlx's own audio loader shells out to ffmpeg directly and
doesn't care about file extension.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Set `SEC_CONTACT`, clone Earnings-21, run the real pipeline

**Files:**
- Modify: `captionlm/config.py:11`
- Modify: `.gitignore` (add `eval_data/`)

**Interfaces:**
- Consumes: everything from Tasks 1-3, plus `captionlm/eval.py`'s
  already-built `run_eval` (unmodified).
- Produces: nothing new — this task runs the pipeline for real and
  reports results. No automated test (live network, live dataset,
  external tools — this IS the verification).

**Context:** This is the payoff task. Nothing here is TDD-shaped; it's
real commands against real infrastructure, run once and checked by
reading their output.

- [ ] **Step 1: Set the real SEC contact**

In `captionlm/config.py`, change:

```python
SEC_CONTACT = "captionlm/1.0 (set a real contact email here)"
```

to:

```python
SEC_CONTACT = "captionlm/1.0 (chielerli@gmail.com)"
```

- [ ] **Step 2: Add `eval_data/` to `.gitignore`**

```bash
echo "eval_data/" >> .gitignore
```

- [ ] **Step 3: Commit the config change**

```bash
git add captionlm/config.py .gitignore
git commit -m "$(cat <<'EOF'
Set a real SEC_CONTACT and gitignore generated eval_data/

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Install git-lfs, local scope only, inside the clone directory**

```bash
mkdir -p vendor_data
cd vendor_data
git clone --filter=blob:none --sparse https://github.com/revdotcom/speech-datasets.git
cd speech-datasets
git lfs install --local
git sparse-checkout set earnings21
```

Expected: `vendor_data/speech-datasets/earnings21/` now contains
`earnings21-file-metadata.csv`, `media/`, `transcripts/nlp_references/`,
`transcripts/wer_tags/`, etc. — pointer files only until the next step
pulls real content.

- [ ] **Step 5: Pull real content for a handful of calls**

The metadata CSV lists calls in no particular order; pull LFS content for
the first 10 `file_id`s (some will be skipped by `assemble_eval_clip` for
resolution failures, so 10 gives `--limit 5` room to actually reach 5):

```bash
cd vendor_data/speech-datasets/earnings21
FILE_IDS=$(tail -n +2 earnings21-file-metadata.csv | head -10 | cut -d, -f1)
for id in $FILE_IDS; do
  git lfs pull --include="media/${id}.mp3"
done
```

Expected: those `.mp3` files are now real audio, not LFS pointer text
(`file media/<id>.mp3` should report an actual audio format, not
"ASCII text").

- [ ] **Step 6: Run `build_eval_set.py`**

From the repo root:

```bash
venv/bin/python -m captionlm.build_eval_set vendor_data/speech-datasets/earnings21 --limit 5
```

Expected: prints one `skip <id>: <reason>` line per call that didn't
resolve (a real possibility — the final whole-branch review flagged that
`filings.recent` may not cover 2020-2021 8-Ks at all; this step is what
actually answers that), and a final `Assembled N clips into
eval_data/clips; terms written to eval_data/terms.txt` line. If `N` is 0,
see the note below before treating it as a failure.

**If zero clips assemble:** check the printed skip reasons. "no CIK
match" means `resolve_cik_from_company_name`'s normalization needs
tuning for that particular company name — expected for some fraction of
real messy names, not a bug to chase for every case. "no 8-K found"
across most/all calls means the final review's pagination concern is
real; that's a genuine finding to report, not silently work around (per
this plan's spec — see Non-goals).

- [ ] **Step 7: Run `eval.py` against the assembled clips**

```bash
venv/bin/python -m captionlm.eval eval_data/clips eval_data/terms.txt
```

Expected: two lines of real numbers, e.g.:
```
baseline: WER=0.XXX P=0.XXX R=0.XXX F=0.XXX
biased: WER=0.XXX P=0.XXX R=0.XXX F=0.XXX
```

Report these numbers — whatever they are — to the user. This is real
data; do not round, cherry-pick, or re-run with different settings to
get a nicer-looking result before reporting.

- [ ] **Step 8: Report the run**

Summarize for the user: how many calls resolved out of how many attempted,
the actual WER/F-score numbers (biasing on vs off), and any systematic
skip reason if one dominated the log from Step 6. No commit for this step
— `eval_data/` is gitignored, and the numbers themselves belong in the
conversation/report, not a file.

---

## Self-Review Notes

- **Spec coverage:** Task 1 covers the `doc_import.py` refactor section.
  Tasks 2-3 cover every function in the spec's Architecture section
  (`extract_call_year`, `estimate_call_date`,
  `resolve_cik_from_company_name`, `reconstruct_transcript_text`,
  `fetch_filing_text`, `convert_to_wav`, `assemble_eval_clip`, `main`).
  Task 4 covers the spec's Scope bullets on `SEC_CONTACT` and running the
  real pipeline. The spec's Non-goals (entity-tag precision/recall
  metric, full-corpus clone, blind-fixing the pagination risk) are
  correctly absent from every task.
- **Placeholder scan:** no TBD/TODO; Task 4's "if zero clips assemble"
  branch is a real, actionable instruction (check skip reasons, report
  honestly), not a placeholder for unhandled failure.
- **Type consistency:** `assemble_eval_clip`'s signature and return type
  (`dict | None` with `file_id`/`terms`/`filing_url` keys) match between
  its definition (Task 3, Step 3) and its test (Task 3, Step 1).
  `find_press_release_filing`'s `window_days` parameter name matches its
  real definition in the already-built `captionlm/eval_dataset.py`
  (verified by reading the file directly during planning, not assumed).
