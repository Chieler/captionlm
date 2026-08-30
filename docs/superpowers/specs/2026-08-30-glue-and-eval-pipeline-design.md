# Glue and Eval Pipeline — Design

Date: 2026-08-30
Status: approved, pending implementation plan

## Problem

The domain-biased captioning tool (`captionlm/`) was built and reviewed
task-by-task in a prior plan (`docs/superpowers/specs/2026-08-29-domain-biased-captioning-design.md`).
The final whole-branch review found that every individual piece works —
term extraction, the context graph, the CTC-WS biased model, the CLI, the
eval harness, SEC EDGAR filing lookup — but nothing chains them together
end to end, and the tool has never been run against real data. `file_import.py`
also still lives at the repo root, outside the `captionlm/` package, as a
script with module-level side effects (it renames files and prints on
import). This spec closes both gaps: fold `file_import.py` into the
package as a real function, add the missing orchestration to assemble a
real eval set from Earnings-21 + SEC EDGAR, and run `eval.py` for real
numbers.

## Scope

- Move `file_import.py` into `captionlm/doc_import.py`, refactored into a
  pure, importable `extract_text(path) -> str` function (no module-level
  side effects, no hardcoded directory scan on import).
- Add `captionlm/build_eval_set.py`: the orchestration that was missing —
  fetch a filing → extract its text → extract terms from it → write the
  `<clip>.wav`/`<clip>.txt`/`<clip>.doc.txt` files `eval.py` and the
  original design's data contract expect.
- Set `captionlm/config.py`'s `SEC_CONTACT` to a real contact
  (`chielerli@gmail.com`, per explicit confirmation — this goes out in
  plain text on every SEC EDGAR request, which is SEC's stated purpose
  for the field).
- Clone a handful (default 5) of Earnings-21 calls, assemble their clips,
  and run `eval.py`'s existing WER + term-F-score harness (biasing on vs
  off) against them for real.

## Non-goals

- The term-extraction-vs-entity-tags precision/recall metric the original
  design also specced — deferred as a follow-up once the entity-tag file
  format is actually known (see Open/known unknowns below).
- Any restructuring beyond moving `file_import.py` — no other file moves,
  splits, or renames.
- Cloning the full Earnings-21 corpus (44 calls, ~39h) — a handful is
  enough to validate the pipeline, per the original design's own
  recommendation.
- Live verification of SEC EDGAR's `filings.recent` pagination limits
  (flagged as a real risk by the final review) — this work will surface
  that risk empirically (assembly either finds a filing or it doesn't;
  failures are logged and skipped, not treated as a blocker to fix
  blindly).

## Architecture

Verified against a real (metadata-only, no audio) clone of
`revdotcom/speech-datasets` before writing this section — see Resolved
unknowns below for what that changed from the original plan.

```
static file, or a downloaded SEC filing (temp file)
        │
        ▼
captionlm/doc_import.py
  extract_text(path: str, max_length: int = 50_000) -> str
        │
        ▼
captionlm/term_extraction.py (unchanged)
  extract_terms(text, top_n) -> list[str]
        │
        ▼
captionlm/build_eval_set.py                          [NEW]
  extract_call_year(nlp_path, wer_tags_path) -> int | None
      parses the .nlp token file + wer_tags sidecar JSON for the first
      token tagged entity_type "YEAR", returns its text as an int

  estimate_call_date(year: int, financial_quarter: int) -> str
      quarter → approximate reporting-month midpoint (Q1→May 15,
      Q2→Aug 15, Q3→Nov 15, Q4→Feb 15 of year+1), "YYYY-MM-DD"

  resolve_cik_from_company_name(company_name, company_tickers: dict) -> str | None
      same shape as eval_dataset.resolve_cik, matches against each
      entry's "title" (not "ticker") after normalizing case/suffixes

  reconstruct_transcript_text(nlp_path: str) -> str
      joins the .nlp file's token+punctuation columns into plain text
      (jiwer's default transform lowercases and strips punctuation on
      both sides anyway, so exact casing/spacing fidelity isn't load-
      bearing here)

  fetch_filing_text(url: str) -> str
      urllib download (SEC_CONTACT User-Agent) → temp .htm file
      → doc_import.extract_text → cleanup temp file

  convert_to_wav(mp3_path: str, wav_path: str) -> None
      ffmpeg subprocess — parakeet-mlx's own load_audio already shells
      out to ffmpeg and doesn't care about file extension, so this
      isn't for the model; it's because eval.py's load_clips globs
      *.wav, and an eval_data/clips/ directory that lies about its own
      file formats is a bad thing to leave behind for later debugging

  assemble_eval_clip(file_id, company_name, financial_quarter,
      audio_path, nlp_path, wer_tags_path, out_dir: str) -> dict | None
      → extract_call_year, estimate_call_date
      → fetch_json(company_tickers) [eval_dataset.py, already lru_cached]
      → resolve_cik_from_company_name
      → eval_dataset.find_press_release_filing(cik, call_date,
            submissions, window_days=60)  [imported directly — see
            "Why not assemble_triple" below]
      → eval_dataset.build_filing_url → fetch_filing_text
      → extract_terms(filing_text) → write <clip>.doc.txt
      → convert_to_wav(audio_path) → <out_dir>/<clip>.wav
      → reconstruct_transcript_text(nlp_path) → <out_dir>/<clip>.txt
      → union of all clips' terms → shared terms.txt for the eval run
      → None (skip + log) at any resolution failure — never guesses

  main()  — CLI: reads earnings21-file-metadata.csv, takes --limit
            (default 5), calls assemble_eval_clip per row, writes
            eval_data/clips/ + eval_data/terms.txt

        │
        ▼
captionlm/eval.py (unchanged)
  run_eval(clip_dir="eval_data/clips", term_list_path="eval_data/terms.txt")
      → real WER + F-score, biasing on vs off
```

### Why not call `eval_dataset.assemble_triple` directly

`assemble_triple` composes `resolve_cik` → `find_press_release_filing`
(hardcoded `window_days=5`) → `build_filing_url`, with no way to pass a
wider window through. Since the estimated call date here is only
approximate (see below), a 5-day window will miss real matches.
Rather than modify `eval_dataset.py` (already task-reviewed as
lookup-only) to add a `window_days` passthrough it didn't need before,
`build_eval_set.py` imports and calls `find_press_release_filing`
directly with `window_days=60`, alongside `resolve_cik_from_company_name`
(a new function, since Earnings-21 gives a company name, not a ticker —
see below) in place of `resolve_cik`. Nothing about `eval_dataset.py`
changes.

### Why a new orchestrator module, not spread across existing files

`eval_dataset.py`, `term_extraction.py`, and the refactored `doc_import.py`
each already passed task-scoped review as single-responsibility, mostly
pure modules (`eval_dataset.py`'s reviewer specifically verified it does
only lookups, no filesystem I/O beyond `fetch_json`). Adding orchestration
to any of them would blur a boundary that already passed review for being
clean. `build_eval_set.py` is the one new module that's allowed to know
about all of them and do real I/O (network fetch, file copies) — one new
file, one new responsibility.

### `doc_import.py` refactor

Current `file_import.py` is a script: it creates an `Extractor` at import
time, lists and renames every file in a hardcoded `static_assets/` path,
and prints results — none of which is safe to import from
`build_eval_set.py`. The refactor:

- `extract_text(path: str, max_length: int = 50_000) -> str` — the actual
  extraction + TIKA-198 retry logic (the `unwrap_google_docs_sdt`/
  `fix_google_docs_docx` helpers move over unchanged as internal
  functions), parameterized on `max_length` instead of a hardcoded 3000
  (that limit was sized for short reference docs; SEC filings and other
  real documents need much more headroom — 50,000 chars is a deliberate,
  generous default, not a hard constraint).
- The old "rename and scan `static_assets/`" behavior moves under
  `if __name__ == "__main__":` for manual/demo use — it does not run on
  import.

### `build_eval_set.py`'s clip output contract

Matches what `eval.py.load_clips` already expects (`<clip>.wav` +
`<clip>.txt` pairs in one directory) plus the original design's
`<clip>.doc.txt` (extracted+term-extracted filing text) for future use by
the deferred entity-tag metric. `eval_data/clips/` is new and gitignored
(generated data, like `vendor_data/` from the prior plan).

### Resolved unknowns — verified against a real (metadata-only) clone

The original version of this spec deferred Earnings-21's exact file
layout to "discover during implementation." That research happened
during planning instead (git-lfs installed local-scope only, in a
throwaway `/tmp` clone — never touched this repo's or the user's git
config), because the shape of that data changes what code needs to
exist, not just its details:

- **No dataset ships an exact call date.** `earnings21-file-metadata.csv`
  has `file_id, company_name, financial_quarter, sector, ...` — no
  ticker, no date. The sibling `earnings22/metadata.csv` has a real
  `Ticker Symbol` column but *also* no date, and skews toward
  non-English/non-US filers (out of scope here). The only real date
  signal anywhere is inside Earnings-21's own transcripts: `.nlp` token
  files carry entity tags (via a `wer_tags` sidecar JSON), and `YEAR`
  is one of them — confirmed on a real file (`4320211`): the token
  `"2020"` is tagged `YEAR`, alongside a `DATE`-tagged span reading "the
  third quarter Fiscal 2020". `extract_call_year` + `estimate_call_date`
  (above) turn that plus the CSV's `financial_quarter` into an
  approximate target date — a real signal from the call's own content,
  not a fabricated one, but approximate enough that it needs the widened
  60-day window rather than the original 5-day default.
- **Transcripts are not plain text.** `.nlp` files are pipe-separated
  token tables (`token|speaker|ts|endTs|punctuation|case|tags|wer_tags`).
  `reconstruct_transcript_text` builds the plain-text reference `eval.py`
  needs from the `token`+`punctuation` columns.
- **Audio is `.mp3`, at various sample rates** (confirmed 24000/44100 Hz
  across different files) — `parakeet-mlx`'s own `load_audio` shells out
  to `ffmpeg -i <path>` directly regardless of extension or rate, so it
  doesn't need conversion. `eval.py.load_clips` globbing `*.wav` is the
  actual reason `convert_to_wav` exists.
- **No ticker, only a company name.** `resolve_cik_from_company_name`
  matches Earnings-21's `company_name` against SEC's `company_tickers.json`
  `"title"` field instead of routing through a ticker symbol at all.

**Whether `filings.recent` covers 2020-2021 8-Ks at all** (flagged by the
final review as a separate risk) is still genuinely unverified — that one
needs a live network call, which wasn't made during this (offline,
metadata-only) research pass. `assemble_eval_clip` treats a resolution
failure at any step as "skip this call, log why," so a systematic
pagination gap will show up as most/all calls being skipped when this
actually runs — which is itself the answer, discovered for real rather
than guessed at.

## Testing

- `doc_import.extract_text`: real automated test using
  `static_assets/Philo_Homes_Organization_Structure_Proposal.docx` (the
  actual Google-Docs-exported file that originally triggered TIKA-198) —
  a genuine regression test for a bug that was already found and fixed
  once.
- `extract_call_year`, `estimate_call_date`, `resolve_cik_from_company_name`,
  `reconstruct_transcript_text`: all pure, local, no network — real
  automated tests against small synthetic fixtures (a 3-line `.nlp` +
  matching `wer_tags.json`, a fixture `company_tickers` dict), same
  pattern already established by `eval_dataset.py`'s own tests. No need
  for the real multi-MB dataset files to test this logic.
- `build_eval_set.assemble_eval_clip`: the network fetch
  (`fetch_filing_text`) is the natural seam to isolate — a test
  monkeypatches it to return fixed filing text, with real temp audio
  (a tiny silent file) and `.nlp`/`wer_tags` fixtures, then verifies the
  clip directory ends up with the right three files in the right place.
- `fetch_filing_text`, `convert_to_wav` (needs real ffmpeg/audio), and
  `build_eval_set.main()` are not covered by an automated test (live
  network, live dataset, external tool) — verified by actually running
  the pipeline against real Earnings-21 calls, which is the point of
  this work.
- No new tests needed for `eval.py` itself — unchanged, already tested.

## Phases

1. **`doc_import.py`** — move + refactor `file_import.py`, add the
   TIKA-198 regression test.
2. **`build_eval_set.py` pure functions** — `extract_call_year`,
   `estimate_call_date`, `resolve_cik_from_company_name`,
   `reconstruct_transcript_text`, each with a fixture-based test.
3. **`build_eval_set.py` orchestration** — `fetch_filing_text`,
   `convert_to_wav`, `assemble_eval_clip` (with its monkeypatched-network
   test), `main()`.
4. **Set `SEC_CONTACT`** in `captionlm/config.py` to
   `chielerli@gmail.com`.
5. **Clone Earnings-21 for real** (git-lfs, local-scope install, a
   handful of calls via sparse-checkout — not the full 39h) and **run
   it**: `build_eval_set.py` against 5 real calls, then `eval.py`
   against the assembled clip set. Report real WER/F-score numbers,
   biasing on vs off.
