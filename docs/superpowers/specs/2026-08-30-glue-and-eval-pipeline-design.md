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
  fetch_filing_text(url: str) -> str
      urllib download (SEC_CONTACT User-Agent) → temp .htm file
      → doc_import.extract_text → cleanup temp file

  assemble_eval_clip(ticker, call_date, audio_path,
      transcript_path, out_dir: str) -> dict | None
      → captionlm.eval_dataset.assemble_triple(ticker, call_date)
      → fetch_filing_text(filing["url"])
      → extract_terms(filing_text) → write <clip>.doc.txt
      → copy audio_path  → <out_dir>/<clip>.wav
      → copy transcript_path → <out_dir>/<clip>.txt
      → union of all clips' terms → shared terms.txt for the eval run

  main()  — CLI: reads Earnings-21's own call metadata (schema TBD,
            see Open/known unknowns), takes --limit (default 5),
            calls assemble_eval_clip per call, writes eval_data/clips/

        │
        ▼
captionlm/eval.py (unchanged)
  run_eval(clip_dir="eval_data/clips", term_list_path="eval_data/terms.txt")
      → real WER + F-score, biasing on vs off
```

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

### Open/known unknowns — resolved during implementation, not guessed here

- **Earnings-21's per-call metadata schema** (ticker, call date per audio
  file) is not yet known — the dataset isn't cloned. First implementation
  step is clone a handful of calls, inspect whatever metadata file
  actually ships, and adapt `build_eval_set.main()`'s call-list loading to
  match. This is expected, ordinary discovery work, not a design risk —
  the pipeline shape (ticker/date in, clip out) doesn't change regardless
  of the exact source format.
- **Whether `filings.recent` covers 2020-2021 8-Ks at all** (flagged by
  the final review as a real risk) is unverified. `assemble_eval_clip`
  treats a `None` from `assemble_triple` as "skip this call, log why" —
  consistent with `eval_dataset.py`'s existing documented behavior — so a
  systematic pagination gap will show up as most/all calls being skipped,
  which is itself the answer to whether Important finding #4 from the
  final review needs a real fix. Not fixing it blind; finding out for
  real.

## Testing

- `doc_import.extract_text`: real automated test using
  `static_assets/Philo_Homes_Organization_Structure_Proposal.docx` (the
  actual Google-Docs-exported file that originally triggered TIKA-198) —
  a genuine regression test for a bug that was already found and fixed
  once.
- `build_eval_set.assemble_eval_clip`: the network fetch
  (`fetch_filing_text`) is the natural seam to isolate — a test
  monkeypatches it to return fixed filing text and real temp audio/text
  files, then verifies the clip directory ends up with the right three
  files in the right place. This matches `eval_dataset.py`'s own
  established pattern of keeping pure/testable logic separate from the
  one function that does real network I/O.
- `fetch_filing_text` and `build_eval_set.main()` are not covered by an
  automated test (live network, live dataset) — verified by actually
  running the pipeline against real Earnings-21 calls, which is the
  point of this work.
- No new tests needed for `eval.py` itself — unchanged, already tested.

## Phases

1. **`doc_import.py`** — move + refactor `file_import.py`, add the
   TIKA-198 regression test.
2. **Clone + inspect Earnings-21** — pull a handful of calls via
   git-lfs, inspect the actual metadata format, confirm the plan for
   Phase 3 against reality (may require a small course-correction here,
   expected).
3. **`build_eval_set.py`** — `fetch_filing_text`, `assemble_eval_clip`
   (with its monkeypatched-network test), `main()`.
4. **Set `SEC_CONTACT`** in `captionlm/config.py` to
   `chielerli@gmail.com`.
5. **Run it** — `build_eval_set.py` against 5 real calls, then
   `eval.py` against the assembled clip set. Report real WER/F-score
   numbers, biasing on vs off.
