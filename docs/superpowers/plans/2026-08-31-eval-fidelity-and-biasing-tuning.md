# Eval Fidelity and Biasing Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the four measurement defects that push the 2026-08-30 eval result toward "biasing hurts", then re-run the sweep to get a trustworthy answer on whether CTC-WS context biasing helps this domain.

**Architecture:** All changes land in the eval harness (`captionlm/eval.py`, `captionlm/build_eval_set.py`) or in one tuned constant in `captionlm/config.py`. The eval currently biases every clip with every other company's terms and runs at NeMo's untuned `cb_weight=3.0`; this plan makes the eval measure the shipped one-document-per-clip configuration and calibrates the weight. The user-facing runtime path (`captionlm/cli.py:31-41`) is not modified.

**Tech Stack:** Python 3.12, existing deps only (`jiwer` 4.0.0, `kaldialign`, `pyate`, `parakeet-mlx`, `sentencepiece`), stdlib `json`/`urllib`/`csv`/`datetime`.

**Spec:** `docs/superpowers/specs/2026-08-31-eval-fidelity-and-biasing-tuning-design.md`

## Global Constraints

- **No change may add work to `captionlm/cli.py`'s path.** The central product requirement is that the user never waits for a model to train. Any task that would make the user wait longer at caption time is out of scope by definition. Verified safe: every task below touches only the eval harness or a `config.py` constant.
- No new dependencies. `requirements.txt` is not modified by this plan.
- Test runner: `venv/bin/pytest`, run from the repo root (`/Users/chielerli/Programming/captionlm`).
- Every commit ends with a `Co-Authored-By: Claude <Model> <noreply@anthropic.com>` trailer naming the model doing the work, matching this repo's existing commits (current commits use `Claude Sonnet 5`).
- Commit message style in this repo: a sentence-case summary line describing the *effect*, then a body paragraph explaining what the review found and what changed. No `feat:`/`fix:` prefixes — this repo does not use them.
- `captionlm/vendor/` holds adapted NeMo code. Do not change algorithms there; only parameters already exposed through `SpotterConfig` may be tuned.

---

## File Structure

| File | Change | Responsibility after this plan |
|---|---|---|
| `captionlm/build_eval_set.py` | Modify | Assembles eval clips from Earnings-21 + EDGAR. Gains: per-clip terms file (T1), EX-99.1 exhibit ranking (T3), EDGAR shard pagination (T4). |
| `captionlm/eval.py` | Modify | Scores biased vs unbiased. Gains: per-clip context graphs, split baseline/biased arms, per-clip JSON output (T2). |
| `captionlm/eval_dataset.py` | Modify | Shrinks to EDGAR HTTP primitives only: `fetch_json`, `build_filing_url`, and the two URL constants (T6). Dead lookup functions deleted. |
| `captionlm/config.py` | Modify | `cb_weight` set to the swept value, docstring updated (T5). |
| `tests/test_build_eval_set.py` | Modify | Adds coverage for T1, T3, T4. |
| `tests/test_eval.py` | Modify | Adds coverage for T2. |
| `tests/test_eval_dataset.py` | Modify | Shrinks to `test_build_filing_url` (T6). |
| `docs/superpowers/specs/2026-08-30-glue-and-eval-pipeline-design.md` | Modify | Corrected: the false jiwer claim and the stale import description (T6). |
| `docs/superpowers/specs/2026-08-31-eval-fidelity-and-biasing-tuning-design.md` | Modify | Gains the sweep results table (T5). |

Tasks 1, 3, 4 all touch `build_eval_set.py` but different functions, and are independently reviewable. Task 2 depends on Task 1's output format. Task 5 depends on 1, 2, 3. Task 6 depends on 5.

---

## Task 1: Per-clip term list written at assembly time

**Files:**
- Modify: `captionlm/build_eval_set.py:199-208` (`assemble_eval_clip`)
- Test: `tests/test_build_eval_set.py`

**Interfaces:**
- Consumes: `write_term_list(terms: list[str], path: str) -> None`, already imported at `build_eval_set.py:33`.
- Produces: `assemble_eval_clip` now writes a fourth file per clip, `<out_dir>/<file_id>.terms.txt`, one term per line. Task 2 reads this filename.

**Context:** `main()` currently unions every clip's terms into a single `eval_data/terms.txt` (`build_eval_set.py:262`), and `eval.py` loads that one file for all clips. With `TERM_EXTRACTION_TOP_N = 50` and 5 clips the context graph carries ~250 terms, ~200 of which belong to other companies and can only ever produce false accepts. Keep writing the combined file — `compute_fscore` must score against the full term vocabulary, and shrinking that would change the metric rather than the method. Only the *context graph* becomes per-clip.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_build_eval_set.py`, after `test_assemble_eval_clip_writes_expected_files`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_build_eval_set.py::test_assemble_eval_clip_writes_per_clip_terms_file -v`

Expected: FAIL with `AssertionError: assert False` on `terms_file.exists()`.

- [ ] **Step 3: Write minimal implementation**

In `captionlm/build_eval_set.py`, inside `assemble_eval_clip`, after the `.doc.txt` write and before the `return`:

```python
    with open(os.path.join(out_dir, f"{file_id}.doc.txt"), "w", encoding="utf-8") as f:
        f.write(filing_text)
    write_term_list(terms, os.path.join(out_dir, f"{file_id}.terms.txt"))

    return {"file_id": file_id, "terms": terms, "filing_url": filing_url}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_build_eval_set.py -v`

Expected: PASS, all tests in the file including the pre-existing ones.

- [ ] **Step 5: Commit**

```bash
git add captionlm/build_eval_set.py tests/test_build_eval_set.py
git commit -m "Write a per-clip term list alongside each assembled eval clip

The eval harness unioned every clip's terms into one file, so each clip
was biased with roughly 200 terms drawn from other companies' filings.
Those terms cannot be spoken in the clip, so every spot they produce is
a false accept. Emit <file_id>.terms.txt per clip so eval.py can build a
graph from just that clip's document. The combined terms.txt is still
written; compute_fscore needs the full vocabulary to score against."
```

**Why this improves WER:** shrinks the context graph from ~250 terms to ~50 per clip, removing a strictly one-directional source of substitution errors, and makes the eval measure the configuration `cli.py` actually ships.

---

## Task 2: Per-clip context graphs, split arms, and persisted results

**Files:**
- Modify: `captionlm/eval.py` (whole file)
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `<clip>.terms.txt` from Task 1; `load_term_list(path) -> list[str]` and `build_context_graph(terms, tokenizer, blank_idx) -> ContextGraphCTC` from `captionlm.terms`; `compute_fscore(samples, key_words_list) -> tuple[float, float, float]` from `captionlm.vendor.fscore`.
- Produces:
  - `load_clips(clip_dir: str, fallback_terms: list[str] | None = None) -> list[dict]` — each dict now has keys `wav`, `text`, `terms`.
  - `transcribe_baseline(model, clips: list[dict]) -> list[dict]`
  - `transcribe_biased(model, clips, tokenizer, blank_idx: int, cb_weight: float | None = None) -> list[dict]`
  - `score(clips, baseline, biased, all_terms) -> dict` with keys `baseline`, `biased`, `per_clip`.
  - `run_eval(clip_dir, term_list_path, model_id=MODEL_ID, cb_weight=None, baseline=None) -> dict` — accepts a precomputed `baseline` to skip the unbiased pass; returns the `score` dict plus `cb_weight` and `baseline_samples`.
  - `main()` gains `--out-json PATH`.

**Context:** The baseline arm does not depend on `cb_weight`, so the Task 5 sweep must not re-transcribe it per point — a 5-point sweep would be 10 transcription passes where 6 suffice, and each pass is roughly 20 minutes per clip on this hardware. `main()` currently prints four floats and writes nothing; the 2026-08-30 run's artifacts were lost with the worktree.

- [ ] **Step 1: Write the failing tests**

Replace the contents of `tests/test_eval.py` with:

```python
import json

from captionlm.eval import _wer, load_clips, score, transcribe_biased


def test_load_clips_pairs_wav_and_txt(tmp_path):
    (tmp_path / "clip1.wav").write_bytes(b"\x00")
    (tmp_path / "clip1.txt").write_text("hello there", encoding="utf-8")
    (tmp_path / "clip2.wav").write_bytes(b"\x00")
    (tmp_path / "clip2.txt").write_text("second clip", encoding="utf-8")
    (tmp_path / "orphan.wav").write_bytes(b"\x00")  # no matching .txt

    clips = load_clips(str(tmp_path))

    assert len(clips) == 2
    texts = {c["text"] for c in clips}
    assert texts == {"hello there", "second clip"}
    assert all(c["wav"].endswith(".wav") for c in clips)


def test_load_clips_reads_per_clip_terms(tmp_path):
    (tmp_path / "clip1.wav").write_bytes(b"\x00")
    (tmp_path / "clip1.txt").write_text("hello there", encoding="utf-8")
    (tmp_path / "clip1.terms.txt").write_text("net sales\nfree cash flow\n", encoding="utf-8")

    clips = load_clips(str(tmp_path), fallback_terms=["ignored"])

    assert clips[0]["terms"] == ["net sales", "free cash flow"]


def test_load_clips_falls_back_when_no_per_clip_terms(tmp_path):
    (tmp_path / "clip1.wav").write_bytes(b"\x00")
    (tmp_path / "clip1.txt").write_text("hello there", encoding="utf-8")

    clips = load_clips(str(tmp_path), fallback_terms=["combined term"])

    assert clips[0]["terms"] == ["combined term"]


def test_transcribe_biased_builds_a_graph_per_clip(monkeypatch):
    graphs_seen = []

    class FakeResult:
        text = "hello"

    class FakeModel:
        context_graph = None

        def transcribe(self, wav_path, chunk_duration=120.0):
            graphs_seen.append(self.context_graph)
            return FakeResult()

    monkeypatch.setattr(
        "captionlm.eval.build_context_graph",
        lambda terms, tokenizer, blank_idx: tuple(terms),
    )

    clips = [
        {"wav": "a.wav", "text": "x", "terms": ["alpha"]},
        {"wav": "b.wav", "text": "y", "terms": ["beta"]},
    ]
    samples = transcribe_biased(FakeModel(), clips, tokenizer=None, blank_idx=0)

    assert graphs_seen == [("alpha",), ("beta",)]
    assert [s["text"] for s in samples] == ["x", "y"]


def test_score_emits_one_record_per_clip():
    clips = [
        {"wav": "/tmp/a.wav", "text": "net sales grew", "terms": ["net sales"]},
        {"wav": "/tmp/b.wav", "text": "free cash flow rose", "terms": ["free cash flow"]},
    ]
    baseline = [
        {"text": "net sales grew", "pred_text": "net sails grew"},
        {"text": "free cash flow rose", "pred_text": "free cash flow rose"},
    ]
    biased = [
        {"text": "net sales grew", "pred_text": "net sales grew"},
        {"text": "free cash flow rose", "pred_text": "free cash flow rose"},
    ]

    stats = score(clips, baseline, biased, ["net sales", "free cash flow"])

    assert [r["clip"] for r in stats["per_clip"]] == ["a.wav", "b.wav"]
    assert stats["per_clip"][0]["baseline_wer"] > 0.0
    assert stats["per_clip"][0]["biased_wer"] == 0.0
    assert stats["biased"]["wer"] < stats["baseline"]["wer"]
    assert stats["per_clip"][0]["n_terms"] == 1


def test_wer_ignores_case_and_punctuation_but_counts_real_errors():
    # Regression: jiwer 4.0.0's default transform does not normalize, and
    # jiwer.wer() takes reference_transform (not truth_transform).
    assert _wer(["Net Sales grew."], ["net sales grew"]) == 0.0
    assert _wer(["net sales grew"], ["net sails grew"]) > 0.0


def test_score_ignores_case_and_punctuation():
    clips = [{"wav": "/tmp/a.wav", "text": "Net Sales grew.", "terms": ["net sales"]}]
    samples = [{"text": "Net Sales grew.", "pred_text": "net sales grew"}]

    stats = score(clips, samples, samples, ["net sales"])

    assert stats["baseline"]["wer"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_eval.py -v`

Expected: FAIL with `ImportError: cannot import name 'score' from 'captionlm.eval'`.

- [ ] **Step 3: Write the implementation**

Replace `captionlm/eval.py` with:

```python
"""Score domain-biased captioning against a labeled clip set: term
F-score and WER, with biasing on vs off. Run captionlm/build_eval_set.py
first to populate clip_dir automatically from Earnings-21, or hand-collect
<clip>.wav/<clip>.txt pairs yourself.

Each clip is biased with its own <clip>.terms.txt when present, matching
what cli.py does: one document biases its own audio. The combined term
list passed on the command line is used for F-score scoring, so the
metric's vocabulary stays constant across clips.
"""
import argparse
import glob
import json
import os

import jiwer

from captionlm.biased_model import load_biased_model
from captionlm.config import MODEL_ID
from captionlm.terms import build_context_graph, load_term_list, load_tokenizer
from captionlm.vendor.fscore import compute_fscore


def load_clips(clip_dir: str, fallback_terms: list[str] | None = None) -> list[dict]:
    clips = []
    for wav_path in sorted(glob.glob(os.path.join(clip_dir, "*.wav"))):
        base = os.path.splitext(wav_path)[0]
        txt_path = base + ".txt"
        if not os.path.isfile(txt_path):
            continue
        with open(txt_path, encoding="utf-8") as f:
            reference = f.read().strip()
        terms_path = base + ".terms.txt"
        terms = load_term_list(terms_path) if os.path.isfile(terms_path) else list(fallback_terms or [])
        clips.append({"wav": wav_path, "text": reference, "terms": terms})
    return clips


def transcribe_clips(model, clips: list[dict]) -> list[dict]:
    samples = []
    for clip in clips:
        result = model.transcribe(clip["wav"], chunk_duration=120.0)
        samples.append({"text": clip["text"], "pred_text": result.text})
    return samples


def transcribe_baseline(model, clips: list[dict]) -> list[dict]:
    model.context_graph = None
    return transcribe_clips(model, clips)


def transcribe_biased(
    model,
    clips: list[dict],
    tokenizer,
    blank_idx: int,
    cb_weight: float | None = None,
) -> list[dict]:
    if cb_weight is not None:
        model.spotter_config.cb_weight = cb_weight
    samples = []
    for clip in clips:
        model.context_graph = build_context_graph(clip["terms"], tokenizer, blank_idx)
        samples.extend(transcribe_clips(model, [clip]))
    return samples


# jiwer 4.0.0's default transform does NOT normalize -- wer_default is
# [RemoveMultipleSpaces, Strip, ReduceToListOfListOfWords], so
# jiwer.wer("Net Sales grew.", "net sales grew") is 1.0. The model emits
# cased, punctuated text and the references are rebuilt from the .nlp
# token+punctuation columns, so without this the score conflates
# recognition errors with casing and punctuation mismatch.
#
# jiwer's own wer_standardize is not the fix: it includes RemoveWhiteSpace,
# which glues every word into a single token. Compose it explicitly.
_NORMALIZE = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def _wer(references: list[str], hypotheses: list[str]) -> float:
    # NB: jiwer.wer takes reference_transform, NOT truth_transform.
    return jiwer.wer(
        references,
        hypotheses,
        reference_transform=_NORMALIZE,
        hypothesis_transform=_NORMALIZE,
    )


def score(clips: list[dict], baseline: list[dict], biased: list[dict], all_terms: list[str]) -> dict:
    references = [c["text"] for c in clips]
    baseline_preds = [s["pred_text"] for s in baseline]
    biased_preds = [s["pred_text"] for s in biased]

    baseline_p, baseline_r, baseline_f = compute_fscore(baseline, all_terms)
    biased_p, biased_r, biased_f = compute_fscore(biased, all_terms)

    per_clip = [
        {
            "clip": os.path.basename(clip["wav"]),
            "reference": clip["text"],
            "baseline_pred": base_pred,
            "biased_pred": bias_pred,
            "baseline_wer": _wer([clip["text"]], [base_pred]),
            "biased_wer": _wer([clip["text"]], [bias_pred]),
            "n_terms": len(clip["terms"]),
        }
        for clip, base_pred, bias_pred in zip(clips, baseline_preds, biased_preds)
    ]

    return {
        "baseline": {
            "wer": _wer(references, baseline_preds),
            "precision": baseline_p,
            "recall": baseline_r,
            "fscore": baseline_f,
        },
        "biased": {
            "wer": _wer(references, biased_preds),
            "precision": biased_p,
            "recall": biased_r,
            "fscore": biased_f,
        },
        "per_clip": per_clip,
    }


def run_eval(
    clip_dir: str,
    term_list_path: str,
    model_id: str = MODEL_ID,
    cb_weight: float | None = None,
    baseline: list[dict] | None = None,
) -> dict:
    all_terms = load_term_list(term_list_path)
    clips = load_clips(clip_dir, fallback_terms=all_terms)
    assert clips, f"no .wav/.txt pairs found in {clip_dir}"

    model = load_biased_model(model_id)
    if baseline is None:
        baseline = transcribe_baseline(model, clips)

    tokenizer = load_tokenizer(model_id)
    biased = transcribe_biased(model, clips, tokenizer, len(model.vocabulary), cb_weight)

    stats = score(clips, baseline, biased, all_terms)
    stats["cb_weight"] = cb_weight
    stats["baseline_samples"] = baseline
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate domain-biased captioning")
    parser.add_argument("clip_dir", help="Directory of <clip>.wav/<clip>.txt pairs")
    parser.add_argument("terms", help="Path to the combined domain term list (.txt), used for F-score scoring")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--cb-weight", type=float, default=None)
    parser.add_argument("--out-json", help="Write full per-clip results to this path as JSON")
    args = parser.parse_args()

    stats = run_eval(args.clip_dir, args.terms, args.model, args.cb_weight)
    for label in ("baseline", "biased"):
        s = stats[label]
        print(f"{label}: WER={s['wer']:.3f} P={s['precision']:.3f} R={s['recall']:.3f} F={s['fscore']:.3f}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in stats.items() if k != "baseline_samples"}, f, indent=2)
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_eval.py -v`

Expected: PASS, 7 tests.

- [ ] **Step 5: Run the full suite for regressions**

Run: `venv/bin/pytest`

Expected: PASS. `main()` no longer iterates `stats.items()` (which would now print the `per_clip` list), so confirm nothing else consumed `run_eval`'s return shape:

Run: `grep -rn "run_eval" captionlm tests --include "*.py"`

Expected: only `captionlm/eval.py` and no test asserting the old two-key shape.

- [ ] **Step 6: Commit**

```bash
git add captionlm/eval.py tests/test_eval.py
git commit -m "Bias each eval clip with its own terms, and persist per-clip results

run_eval built one context graph from the unioned term list and used it
for every clip, so each clip carried roughly 200 terms from other
companies that could only ever false-accept. Build the graph per clip
from <clip>.terms.txt instead, matching what cli.py ships. Split the
baseline and biased arms so a cb_weight sweep can reuse one unbiased
pass instead of recomputing it per point, and add --out-json so a
multi-hour run leaves more behind than four printed floats."
```

**Why this improves WER:** removes the cross-clip term pollution that Task 1 made avoidable, and makes the Task 5 sweep affordable and auditable.

---

## Task 3: Prefer EX-99.1 over other EX-99.x exhibits

**Files:**
- Modify: `captionlm/build_eval_set.py:132-140` (`find_exhibit_document`)
- Test: `tests/test_build_eval_set.py`

**Interfaces:**
- Consumes: `fetch_json(url) -> dict` from `captionlm.eval_dataset`, already imported.
- Produces: `find_exhibit_document(cik: str, accession: str) -> str | None` — same signature, now returns the best-ranked `ex99` document rather than the first one in `index.json` order.

**Context:** On an earnings 8-K, EX-99.1 is the press release — continuous prose the executives read from and paraphrase aloud. EX-99.2 is usually the slide deck or supplemental financial tables. Running `pyate` over a table dump yields column headers and row labels ("Three Months Ended", "Diluted EPS") — formatted text nobody says on the call, so those terms can only false-accept. The current code takes whichever comes first in the directory listing.

- [ ] **Step 1: Write the failing tests**

Add `find_exhibit_document` to the imports at the top of `tests/test_build_eval_set.py`:

```python
from captionlm.build_eval_set import (
    assemble_eval_clip,
    estimate_call_date,
    extract_call_year,
    find_exhibit_document,
    reconstruct_transcript_text,
    resolve_cik_from_company_name,
)
```

Then append:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_build_eval_set.py -k find_exhibit_document -v`

Expected: the first two FAIL with `assert 'ex99-2.htm' == 'ex99-1.htm'`; the last two PASS (current behaviour already satisfies them).

- [ ] **Step 3: Write the implementation**

Replace `find_exhibit_document` in `captionlm/build_eval_set.py`:

```python
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


def find_exhibit_document(cik: str, accession: str) -> str | None:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_build_eval_set.py -v`

Expected: PASS, including the pre-existing `test_assemble_eval_clip_writes_expected_files` whose fixture uses `ex991-pr.htm` (rank 0, still returned).

- [ ] **Step 5: Commit**

```bash
git add captionlm/build_eval_set.py tests/test_build_eval_set.py
git commit -m "Pick the EX-99.1 press release over other EX-99.x exhibits

find_exhibit_document returned whichever ex99 document index.json
listed first, so an 8-K filing its slide deck as EX-99.2 ahead of the
release could hand term extraction a table dump. Column headers and row
labels are never spoken on the call, so those terms can only produce
false accepts in the spotter. Rank exact EX-99.1 variants first and fall
back to any other ex99 when there is no .1."
```

**Why this improves WER:** raises the fraction of graph terms that are actually utterable, cutting dead weight that can only false-accept.

---

## Task 4: Paginate EDGAR's `filings.files[]` shards

**Files:**
- Modify: `captionlm/build_eval_set.py:23` (import) and `106-129` (`find_earnings_release_filing`)
- Test: `tests/test_build_eval_set.py`

**Interfaces:**
- Consumes: `fetch_json(url) -> dict` from `captionlm.eval_dataset`, already imported.
- Produces:
  - `iter_filing_blocks(submissions: dict, call_date: str, window_days: int)` — generator yielding filings blocks (dicts with parallel `form`/`filingDate`/`accessionNumber`/`items` arrays), starting with `filings.recent` and then each in-window shard from `filings.files[]`.
  - `find_earnings_release_filing(cik, call_date, submissions, window_days=60) -> dict | None` — unchanged signature and return shape `{"cik": ..., "accession": ...}`.

**Context:** `submissions.json` caps `filings.recent` at roughly 1000 entries; older filings live in `filings.files[]`, each entry naming a shard JSON with `filingFrom`/`filingTo` bounds. Earnings-21 covers 2020–2021, and for a high-volume filer 1000 filings is well under two years. SEC's shard files carry the column arrays at the JSON root, not nested under `filings.recent`. Only fetch shards whose date range overlaps the search window — do not download a filer's whole history.

- [ ] **Step 1: Write the failing tests**

Add `find_earnings_release_filing` to the imports in `tests/test_build_eval_set.py`, then append:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_build_eval_set.py -k find_earnings_release_filing -v`

Expected: the first two FAIL with `assert None == {'cik': ...}` and `assert 0 == 1`; the third PASSES (current code never fetches).

- [ ] **Step 3: Write the implementation**

Change the import at `captionlm/build_eval_set.py:23`:

```python
from datetime import date, timedelta
```

Then replace `find_earnings_release_filing`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_build_eval_set.py -v`

Expected: PASS, including the pre-existing `test_assemble_eval_clip_skips_when_no_item_202_8k`, whose fixture has no `files` key.

- [ ] **Step 5: Run the full suite**

Run: `venv/bin/pytest`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add captionlm/build_eval_set.py tests/test_build_eval_set.py
git commit -m "Search EDGAR's paginated filing shards, not just filings.recent

submissions.json caps filings.recent at roughly 1000 entries and moves
older ones into filings.files[]. Earnings-21 covers 2020-2021, so for a
high-volume filer the earnings 8-K can sit outside recent entirely and
the clip silently fails to assemble. Walk the shards too, fetching only
those whose filingFrom/filingTo range overlaps the search window."
```

**Why this matters:** does not change per-clip WER. It raises the assembled clip count from 2 toward 15–20, which is the difference between a 0.002 delta indistinguishable from noise and a result worth believing.

---

## Task 5: Run the `cb_weight` calibration sweep

**Depends on:** Tasks 1, 2, 3. Do not start earlier — sweeping against a polluted 250-term graph tunes around the Task 1 bug instead of fixing it.

**Files:**
- Modify: `captionlm/config.py:20` (the `cb_weight` value and the `SpotterConfig` docstring)
- Modify: `docs/superpowers/specs/2026-08-31-eval-fidelity-and-biasing-tuning-design.md` (add the results table)
- Create: `eval_results/sweep-<cb_weight>.json` (gitignored working files)

**Interfaces:**
- Consumes: `run_eval(clip_dir, term_list_path, model_id, cb_weight, baseline) -> dict` from Task 2.
- Produces: a tuned `SpotterConfig.cb_weight` default that `cli.py` inherits with zero added runtime cost.

**Context:** `vendor/ctc_word_spotter.py:326` adds `cb_weight` once per non-blank token, so a 5-token term like `"net sales"` accrues up to +15.0 of pure bonus. `filter_wb_hyps` (`vendor/ctc_word_spotter.py:218-248`) is the false-accept guard: it compares the spot's score against the CTC alignment score over the same frames and drops the spot if the alignment wins. The spot's score carries the accumulated bonus; the alignment's does not. At `cb_weight=3.0` the comparison is rigged by +3 per token in the keyword's favour, and it fails hardest on exactly the long multi-word domain phrases term extraction produces. That is the mechanical cause of the observed precision −0.089 with recall +0.003.

**This task involves multi-hour transcription runs.** Each sweep point transcribes every clip once. Budget accordingly and run points sequentially.

- [ ] **Step 1: Rebuild the eval set with Tasks 3 and 4 in place**

Run:

```bash
venv/bin/python -m captionlm.build_eval_set <path-to-earnings21> \
  --out eval_data/clips --terms-out eval_data/terms.txt --limit 12 --max-attempts 40
```

Expected: at least 10 clips assembled. Each clip directory entry should now have four files (`.wav`, `.txt`, `.doc.txt`, `.terms.txt`).

Verify:

```bash
ls eval_data/clips/*.terms.txt | wc -l
```

If fewer than 10 assembled, read the skip lines — they now carry exception types from the `5d49542` follow-up fix — and record the reasons in the spec before continuing. Do not proceed on fewer than 5 clips; the result will not be interpretable.

- [ ] **Step 2: Write the sweep driver**

Create `scripts/sweep_cb_weight.py`:

```python
"""One-off cb_weight calibration sweep. Transcribes the unbiased baseline
once and reuses it across every sweep point -- the baseline does not
depend on cb_weight, so recomputing it per point doubles the runtime for
nothing."""
import json
import os
import sys

from captionlm.biased_model import load_biased_model
from captionlm.config import MODEL_ID
from captionlm.eval import load_clips, run_eval, transcribe_baseline
from captionlm.terms import load_term_list

CLIP_DIR = "eval_data/clips"
TERMS = "eval_data/terms.txt"
OUT_DIR = "eval_results"
WEIGHTS = [0.5, 1.0, 1.5, 2.0, 3.0]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_terms = load_term_list(TERMS)
    clips = load_clips(CLIP_DIR, fallback_terms=all_terms)
    print(f"{len(clips)} clips, {len(all_terms)} combined terms")

    model = load_biased_model(MODEL_ID)
    print("transcribing baseline once...", flush=True)
    baseline = transcribe_baseline(model, clips)

    for weight in WEIGHTS:
        print(f"cb_weight={weight}...", flush=True)
        stats = run_eval(CLIP_DIR, TERMS, MODEL_ID, cb_weight=weight, baseline=baseline)
        out_path = os.path.join(OUT_DIR, f"sweep-{weight}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in stats.items() if k != "baseline_samples"}, f, indent=2)
        b, x = stats["baseline"], stats["biased"]
        print(
            f"  baseline WER={b['wer']:.3f} F={b['fscore']:.3f} | "
            f"biased WER={x['wer']:.3f} P={x['precision']:.3f} "
            f"R={x['recall']:.3f} F={x['fscore']:.3f}"
        )


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Add the results directory to .gitignore**

Append to `.gitignore`:

```
eval_results/
```

- [ ] **Step 4: Run the sweep**

Run: `venv/bin/python scripts/sweep_cb_weight.py 2>&1 | tee eval_results/sweep.log`

Expected: one line per weight, five `eval_results/sweep-*.json` files.

- [ ] **Step 5: Record the results in the spec**

Add to `docs/superpowers/specs/2026-08-31-eval-fidelity-and-biasing-tuning-design.md`, under a new `## Sweep Results` heading, filling in the real measured numbers:

```markdown
## Sweep Results

Run <DATE>, <N> clips, per-clip term lists, EX-99.1 exhibits.
Baseline (unbiased): WER=<...> P=<...> R=<...> F=<...>

| cb_weight | WER | P | R | F |
|---|---|---|---|---|
| 0.5 | | | | |
| 1.0 | | | | |
| 1.5 | | | | |
| 2.0 | | | | |
| 3.0 | | | | |

Selected: cb_weight=<...>, the WER-minimising point.
```

- [ ] **Step 6: Set the tuned default**

In `captionlm/config.py`, set `cb_weight` to the WER-minimising value and replace the docstring:

```python
@dataclass
class SpotterConfig:
    """CTC-WS hyperparameters. cb_weight was tuned on <N> Earnings-21
    clips on <DATE> (see docs/superpowers/specs/2026-08-31-eval-fidelity-and-biasing-tuning-design.md);
    the other values are NeMo's run_word_spotter defaults and are
    untuned. Note that cb_weight is applied once per non-blank token in
    ctc_word_spotter.py, so its effective magnitude scales with term
    length."""

    beam_threshold: float = 5.0
    cb_weight: float = <TUNED VALUE>
    ctc_ali_token_weight: float = 0.5
    keyword_threshold: float = -5.0
    blank_threshold: float = 0.8
    non_blank_threshold: float = 0.001
    intersection_threshold: float = 30.0
```

- [ ] **Step 7: If no point beats baseline WER, record that instead**

Do **not** tune `keyword_threshold` or `intersection_threshold` to rescue the result — that is a separate experiment and would be fitting to the eval set. Write the negative finding into the spec's Sweep Results section, leave `cb_weight` at the best of the swept values, and note that the honest conclusion is now a claim about the method rather than about an untuned constant.

- [ ] **Step 8: Run the full suite**

Run: `venv/bin/pytest`

Expected: PASS. No test asserts a specific `cb_weight` value.

- [ ] **Step 9: Commit**

```bash
git add captionlm/config.py .gitignore scripts/sweep_cb_weight.py \
  docs/superpowers/specs/2026-08-31-eval-fidelity-and-biasing-tuning-design.md
git commit -m "Tune cb_weight against Earnings-21 instead of shipping NeMo's default

cb_weight is added once per non-blank token in the spotter, so a
five-token term accrues fifteen points of bonus that filter_wb_hyps then
compares against a CTC alignment score carrying none. At NeMo's default
of 3.0 the false-accept guard is rigged in the keyword's favour, hardest
on exactly the long multi-word phrases term extraction produces, which
is why the first run lost nine points of precision for three
thousandths of recall. Sweep the weight and ship the measured value."
```

**Why this improves WER:** restores `filter_wb_hyps` to working order. This is the single largest WER lever in the plan. **User cost: zero** — the sweep is offline and developer-side, and the result ships as a constant.

---

## Task 6: Delete dead lookup code and correct the stale spec

**Depends on:** Task 5, so the spec's numbers can be corrected in one edit.

**Files:**
- Modify: `captionlm/eval_dataset.py` — delete `resolve_cik` (lines 20-25), `find_press_release_filing` (28-49), `assemble_triple` (65-77)
- Modify: `tests/test_eval_dataset.py` — delete the four tests covering them
- Modify: `docs/superpowers/specs/2026-08-30-glue-and-eval-pipeline-design.md`

**Interfaces:**
- Produces: `captionlm/eval_dataset.py` shrinks to `COMPANY_TICKERS_URL`, `SUBMISSIONS_URL_TEMPLATE`, `build_filing_url`, `fetch_json` — the four names `build_eval_set.py:27-32` actually imports.

**Context:** `resolve_cik`, `find_press_release_filing`, and `assemble_triple` have zero production callers — only their own tests. `build_eval_set.find_earnings_release_filing` is a near-copy of `find_press_release_filing` plus the Item 2.02 filter. Separately, the 2026-08-30 design spec asserts that "jiwer's default transform lowercases and strips punctuation on both sides anyway". That is false for jiwer 4.0.0, where `wer_default` is `[RemoveMultipleSpaces, Strip, ReduceToListOfListOfWords]` — and the Task-2 review cited that claim as its "cost if wrong is zero" justification for parking a real case-handling finding.

- [ ] **Step 1: Confirm the functions have no callers**

Run:

```bash
grep -rn "resolve_cik\b\|find_press_release_filing\|assemble_triple" captionlm tests --include "*.py"
```

Expected: matches only in `captionlm/eval_dataset.py` and `tests/test_eval_dataset.py`. `resolve_cik_from_company_name` in `build_eval_set.py` is a different name and must not match the `resolve_cik\b` word-boundary pattern — verify it does not appear.

- [ ] **Step 2: Delete the dead functions**

In `captionlm/eval_dataset.py`, delete `resolve_cik`, `find_press_release_filing`, and `assemble_triple`. Remove the now-unused `from datetime import date` import. Update the module docstring:

```python
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
```

- [ ] **Step 3: Delete the orphaned tests**

Replace `tests/test_eval_dataset.py` with:

```python
from captionlm.eval_dataset import build_filing_url


def test_build_filing_url():
    url = build_filing_url("0000320193", "0001-21-000002", "pressrelease.htm")
    assert url == "https://www.sec.gov/Archives/edgar/data/320193/000121000002/pressrelease.htm"
```

- [ ] **Step 4: Run the full suite**

Run: `venv/bin/pytest`

Expected: PASS, with four fewer tests than before this task.

- [ ] **Step 5: Correct the false jiwer claim in the 2026-08-30 spec**

Find the sentence asserting that jiwer's default transform lowercases and strips punctuation. Replace it with:

```markdown
**Correction (2026-08-31):** this is false for jiwer 4.0.0.
`jiwer.wer_default` is `[RemoveMultipleSpaces, Strip,
ReduceToListOfListOfWords]` — no lowercasing, no punctuation stripping.
`jiwer.wer("Net Sales grew.", "net sales grew")` returns 1.0. The model
emits cased, punctuated text and the references are rebuilt from the
`.nlp` token+punctuation columns, so the first run's reported WER
conflated recognition errors with casing and punctuation mismatch.
`eval.py` now passes `wer_standardize` on both sides explicitly.

The Task-2 review cited this claim as its "cost if wrong is zero"
justification for parking the `case`-column finding. That justification
does not hold, and the parked finding should be re-examined on its
merits.
```

- [ ] **Step 6: Correct the stale import description**

Find the text describing `build_eval_set` importing `eval_dataset.find_press_release_filing` with `window_days=60`. Replace with:

```markdown
**Correction (2026-08-31):** `5d49542` moved this logic into
`build_eval_set.find_earnings_release_filing`, which adds the Item 2.02
filter that `find_press_release_filing` lacked. `eval_dataset.py` now
exports only the EDGAR HTTP primitives (`fetch_json`,
`build_filing_url`, and the two URL constants); its lookup functions
were deleted as dead code.
```

- [ ] **Step 7: Record that the two runs are not comparable**

Add to the same spec, near the results:

```markdown
**Correction (2026-08-31):** the 2026-08-30 run produced two result sets
that are not comparable. The first (WER 0.263 baseline / 0.268 biased)
used 5 clips whose terms came from 8-K cover pages and unrelated
filings, and is invalid. The second (0.260 / 0.262) used 2 clips with
real EX-99.1 text. Different clip sets and different sample sizes: the
0.263 to 0.260 movement is not an improvement, it is a different
experiment.
```

- [ ] **Step 8: Commit**

```bash
git add captionlm/eval_dataset.py tests/test_eval_dataset.py \
  docs/superpowers/specs/2026-08-30-glue-and-eval-pipeline-design.md
git commit -m "Delete dead EDGAR lookup code and correct the stale design spec

resolve_cik, find_press_release_filing and assemble_triple had no
production callers after 5d49542 moved filing lookup into
build_eval_set.py with an Item 2.02 filter; eval_dataset.py is now just
the EDGAR HTTP primitives. The design spec also claimed jiwer's default
transform lowercases and strips punctuation, which is false for jiwer
4.0.0 -- and that claim was used to justify parking a real
case-handling finding, so correct it in place rather than leaving the
next reader to re-derive it."
```

**Why this improves WER:** it does not. Zero runtime impact. It is in the plan because the false premise has already been used once to dismiss a genuine bug.

---

## Expected Outcome

After Tasks 1–5, the biased arm should show WER at or below baseline and precision recovered toward the 0.938 baseline level, measured on ≥10 clips rather than 2.

If it does not, the plan still succeeds. It converts "biasing hurt at an untuned operating point on a polluted 2-clip eval" into a defensible statement about the method itself, with per-clip JSON to interrogate. That is the actual deliverable.

## Verification Checklist

- [ ] `venv/bin/pytest` green after every task.
- [ ] `git diff master --stat -- captionlm/cli.py` is empty — the user runtime path is untouched, so the no-training constraint holds by construction.
- [ ] `grep -n "cb_weight" captionlm/config.py` shows the swept value, not 3.0.
- [ ] `ls eval_data/clips/*.terms.txt` shows one per assembled clip.
- [ ] `eval_results/sweep-*.json` exists for every swept weight.
