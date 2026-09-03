# Headroom Signal Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the model's own unbiased decode confidence at a candidate term's span predicts whether biasing that term would help or hurt — retrospectively, against Earnings-21 per-clip data already swept — before any of it goes near production.

**Architecture:** For each term in a clip's bias list, run a from-scratch reimplementation of the CTC word spotter's raw beam search (`cb_weight=0`, no boost) to find its best-matching unbiased span, then compare that span's score against what the argmax-decoded path already believes is there (`get_ctc_word_alignment`, already vendored and already trusted for this exact comparison inside `filter_wb_hyps`). The gap between the two is the headroom score. Cross-reference against each term's actual fate (recovered / injected / neither) from the already-completed corrected sweep (`data/eval_results/sweep_earnings21_perclip_v2/`) to see whether the score actually discriminates.

**Tech Stack:** Python 3.12, numpy, the vendored `captionlm/vendor/ctc_word_spotter.py` primitives (`Token`, `WSHyp`, `beam_pruning`, `state_pruning`, `get_ctc_word_alignment`), MLX only for the one extra transcription pass that captures logits. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-03-headroom-signal-validation-design.md`

## Global Constraints

- **This is a measurement, not a feature.** No change to `cli.py`, `select_cb_weight()`, or any production decode path. The only production file touched is `captionlm/biased_model.py`, and only to add an opt-in capture hook that is a no-op unless explicitly enabled.
- **Do not modify `captionlm/vendor/ctc_word_spotter.py`.** It's vendored NeMo code with its own license header. Where its logic is needed but not exposed in the right shape, reimplement fresh (the precedent `captionlm/merge.py` already set), reusing its exported primitives (`Token`, `WSHyp`, `beam_pruning`, `state_pruning`, `get_ctc_word_alignment`) rather than duplicating those.
- **Validate against `data/eval_results/sweep_earnings21_perclip_v2/`**, not the older `sweep_earnings21_perclip/` — the original run was confounded by a term-extraction bug (`docs/results/2026-09-02-per-clip-also-regresses.md`, corrected in `docs/results/2026-09-03-fixing-term-extraction-flips-per-clips-net-positive.md`).
- Run tests with `venv/bin/python -m pytest -q`. The package is installed editable, so no `PYTHONPATH=` prefix.

## File Structure

| File | Responsibility |
|---|---|
| `captionlm/headroom.py` (create) | `raw_spot`, `argmax_span_score`, `unbiased_headroom` — pure numpy + vendored primitives, no model/audio I/O. |
| `tests/test_headroom.py` (create) | Unit tests for all three, hand-built logprobs and context graphs, no model or GPU. |
| `captionlm/biased_model.py` (modify) | Add the opt-in `capture_logits` hook to `BiasedParakeetTDTCTC`. |
| `tests/test_biased_model.py` (modify) | One more GPU-backed test, matching that file's existing convention. |
| `docs/HANDOFF.md` (modify) | "exactly three failures" → "exactly four" (the hook adds a 4th GPU test to that file). |
| `scripts/measure_headroom_signal.py` (create) | Orchestration: captures logits for already-swept clips, computes headroom scores, cross-references against the corrected sweep's recorded outcomes, reports whether the score discriminates. No dedicated pytest coverage, matching `scripts/sweep_cb_weight.py`'s own precedent. |

---

### Task 1: `captionlm/headroom.py` — the headroom score

**Files:**
- Create: `captionlm/headroom.py`
- Test: `tests/test_headroom.py`

**Interfaces:**
- Consumes: `Token`, `WSHyp`, `beam_pruning`, `state_pruning`, `get_ctc_word_alignment` from `captionlm.vendor.ctc_word_spotter`; `ContextGraphCTC` from `captionlm.vendor.context_graph_ctc` (test fixtures only — the module itself takes an already-built graph).
- Produces: `raw_spot(logprobs, context_graph, blank_idx, cb_weight=0.0, beam_threshold=5.0, keyword_threshold=-5.0, blank_threshold=0.8, non_blank_threshold=0.001) -> list[WSHyp]`; `argmax_span_score(word_alignment: list[tuple], start_frame: int, end_frame: int) -> float`; `unbiased_headroom(logprobs, context_graph, asr_model, blank_idx, ctc_ali_token_weight=0.5) -> dict[str, float]`. Task 3 calls `unbiased_headroom` directly and only needs that one function; the other two are internal building blocks it also imports for its own tests.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_headroom.py`:

```python
"""headroom.py against hand-built logprobs -- no model, no GPU, no audio.

Every fixture here was verified by direct execution against the real
vendored functions before being written down (not hand-traced), because
run_word_spotter's beam-search interactions are easy to get subtly wrong
by inspection alone.
"""
import numpy as np

from captionlm.headroom import argmax_span_score, raw_spot, unbiased_headroom
from captionlm.vendor.context_graph_ctc import ContextGraphCTC


class FakeModel:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary


def test_raw_spot_finds_strong_single_token_term():
    # vocab: 0="▁hi" (word-start piece), 1="lo" (continuation piece,
    # irrelevant to this graph), blank=2
    logprobs = np.full((4, 3), -50.0, dtype=np.float32)
    logprobs[0, 0] = -0.1
    logprobs[1, 2] = -0.1
    logprobs[2, 1] = -0.1
    logprobs[3, 2] = -0.1
    graph = ContextGraphCTC(blank_id=2)
    graph.add_to_graph([("hi", [[0]])])

    hyps = raw_spot(logprobs, graph, blank_idx=2, cb_weight=0.0)

    assert len(hyps) == 1
    assert hyps[0].word == "hi"
    assert hyps[0].start_frame == 0
    assert hyps[0].end_frame == 0
    assert hyps[0].score == pytest.approx(-0.1, abs=1e-4)


def test_raw_spot_returns_empty_when_blank_dominates_everywhere():
    logprobs = np.full((2, 3), -50.0, dtype=np.float32)
    logprobs[:, 2] = -0.01  # blank near-certain every frame
    graph = ContextGraphCTC(blank_id=2)
    graph.add_to_graph([("hi", [[0]])])

    assert raw_spot(logprobs, graph, blank_idx=2, cb_weight=0.0) == []


def test_argmax_span_score_single_full_overlap_returns_that_words_score():
    word_alignment = [("cat", 5, 9, 2.5)]
    assert argmax_span_score(word_alignment, 5, 9) == 2.5


def test_argmax_span_score_two_word_overlap_is_weighted_not_summed():
    # First overlapping word counts in full; the second is weighted by
    # its overlap percentage. A plain sum would give 3.0, not ~2.33.
    word_alignment = [("the", 0, 2, 1.0), ("cat", 3, 5, 2.0)]
    result = argmax_span_score(word_alignment, 2, 4)
    assert result == pytest.approx(1.0 + (2 / 3) * 2.0)
    assert result != pytest.approx(3.0)


def test_unbiased_headroom_positive_gap_when_spotted_term_beats_a_weak_competing_word():
    # vocab: 0="▁spot" (the term), 1="▁weak1", 2="weak2" (weak1+weak2
    # merge into one low-confidence argmax word spanning frames 0-1).
    # At frame0, "spot" is a real second-best acoustic candidate (-4.0,
    # clears every pruning threshold) even though "weak1" (-3.0) wins
    # argmax there. The merged "weak1weak2" argmax word's own accumulated
    # score (-5.0, two weak frames) is worse than "spot"'s single-frame
    # score, because raw_spot never has to pay for weak2's bad frame.
    model = FakeModel(["▁spot", "▁weak1", "weak2"])
    logprobs = np.full((2, 4), -30.0, dtype=np.float32)
    logprobs[0] = [-4.0, -3.0, -30.0, -5.0]
    logprobs[1] = [-30.0, -30.0, -3.0, -5.0]
    graph = ContextGraphCTC(blank_id=3)
    graph.add_to_graph([("spot", [[0]])])

    result = unbiased_headroom(context_graph=graph, logprobs=logprobs, asr_model=model,
                                blank_idx=3, ctc_ali_token_weight=0.5)

    assert result == {"spot": pytest.approx(1.0)}


def test_unbiased_headroom_has_no_entry_for_a_term_with_no_acoustic_support():
    model = FakeModel(["▁spot"])
    logprobs = np.full((2, 2), -50.0, dtype=np.float32)
    logprobs[:, 1] = -0.01  # blank near-certain -- spot never clears the gate
    graph = ContextGraphCTC(blank_id=1)
    graph.add_to_graph([("spot", [[0]])])

    result = unbiased_headroom(graph, logprobs, model, blank_idx=1)

    assert "spot" not in result


def test_unbiased_headroom_keeps_the_max_across_two_occurrences():
    # "spot" is spotted twice: at frame0 with a +1.0 gap (beats a weak
    # competing word, same construction as the positive-gap test above)
    # and at frame4 with a -0.5 gap (matches argmax's own reading there
    # exactly -- always exactly -ctc_ali_token_weight when spotter and
    # argmax agree, since argmax's score there is spot's own logprob plus
    # that constant). The max must be +1.0, not the average or the last.
    model = FakeModel(["▁spot", "▁weak1", "weak2"])
    logprobs = np.full((5, 4), -30.0, dtype=np.float32)
    logprobs[0] = [-4.0, -3.0, -30.0, -5.0]
    logprobs[1] = [-30.0, -30.0, -3.0, -5.0]
    logprobs[2] = [-30.0, -30.0, -30.0, -0.01]
    logprobs[3] = [-30.0, -30.0, -30.0, -0.01]
    logprobs[4] = [-0.5, -30.0, -30.0, -5.0]
    graph = ContextGraphCTC(blank_id=3)
    graph.add_to_graph([("spot", [[0]])])

    result = unbiased_headroom(graph, logprobs, model, blank_idx=3, ctc_ali_token_weight=0.5)

    assert result == {"spot": pytest.approx(1.0)}
```

Add `import pytest` at the top of the file alongside `import numpy as np`.

- [ ] **Step 2: Run the tests and confirm they fail on import**

Run: `venv/bin/python -m pytest tests/test_headroom.py -v`
Expected: `ModuleNotFoundError: No module named 'captionlm.headroom'` (or `ImportError` for the three names) — nothing in `captionlm/headroom.py` exists yet.

- [ ] **Step 3: Write the implementation**

Create `captionlm/headroom.py`:

```python
"""Does the model's own unbiased decode confidence at a candidate term's
span predict whether biasing it would help or hurt? This module computes
that signal; it does not decide anything or touch any production path --
see docs/superpowers/specs/2026-09-03-headroom-signal-validation-design.md.
"""
import numpy as np

from captionlm.vendor.context_graph_ctc import ContextGraphCTC
from captionlm.vendor.ctc_word_spotter import (
    Token,
    WSHyp,
    beam_pruning,
    get_ctc_word_alignment,
    state_pruning,
)


def raw_spot(
    logprobs: np.ndarray,
    context_graph: ContextGraphCTC,
    blank_idx: int,
    cb_weight: float = 0.0,
    beam_threshold: float = 5.0,
    keyword_threshold: float = -5.0,
    blank_threshold: float = 0.8,
    non_blank_threshold: float = 0.001,
) -> list[WSHyp]:
    """The token-passing beam search from vendor/ctc_word_spotter.py's
    run_word_spotter, unmodified, stopping before that function's own
    trailing find_best_hyps + filter_wb_hyps calls.

    run_word_spotter itself cannot be reused here: it already runs the
    exact argmax comparison this module computes on its own (as
    filter_wb_hyps), but only as a binary keep/drop, and returns only
    what survives. At cb_weight=0 that survival bar is far too strict to
    be a usable signal -- a candidate only wins outright against the
    argmax path with zero boost helping it, which this module needs the
    graded version of, not the pass/fail.
    """
    start_state = context_graph.root
    active_tokens: list[Token] = []
    next_tokens: list[Token] = []
    spotted_words: list[WSHyp] = []
    blank_threshold = np.log(blank_threshold)
    non_blank_threshold = np.log(non_blank_threshold)

    for frame in range(logprobs.shape[0]):
        active_tokens.append(Token(start_state, start_frame=frame))
        best_score = None
        for token in active_tokens:
            if token.state is context_graph.root and logprobs[frame][blank_idx] > blank_threshold:
                continue
            for transition_state in token.state.next:
                if (
                    token.state is context_graph.root
                    and logprobs[frame][int(transition_state)] < non_blank_threshold
                ):
                    continue
                if transition_state != blank_idx:
                    current_score = token.score + logprobs[frame][int(transition_state)].item() + cb_weight
                else:
                    current_score = token.score + logprobs[frame][int(transition_state)].item()
                if not best_score:
                    best_score = current_score
                else:
                    if current_score < best_score - beam_threshold:
                        continue
                    elif current_score > best_score:
                        best_score = current_score
                new_token = Token(token.state.next[transition_state], current_score, token.start_frame)
                if new_token.state.is_end and new_token.score > keyword_threshold:
                    spotted_words.append(
                        WSHyp(
                            word=new_token.state.word,
                            score=new_token.score,
                            start_frame=new_token.start_frame,
                            end_frame=frame,
                        )
                    )
                    if len(new_token.state.next) == 1:
                        if current_score is best_score:
                            best_score = None
                        continue
                next_tokens.append(new_token)
        next_tokens = beam_pruning(next_tokens, beam_threshold)
        next_tokens = state_pruning(next_tokens)
        active_tokens = next_tokens
        next_tokens = []
    return spotted_words


def argmax_span_score(word_alignment: list[tuple], start_frame: int, end_frame: int) -> float:
    """Overlap-weighted sum of get_ctc_word_alignment's word scores across
    [start_frame, end_frame]. Same accumulation vendor/ctc_word_spotter.py's
    filter_wb_hyps does internally (its overall_spot_score) -- the first
    overlapping word counts in full, each subsequent overlapping word is
    weighted by its overlap percentage -- factored out here as a graded
    value because filter_wb_hyps only returns a survive/drop decision.
    """
    hyp_interval = set(range(start_frame, end_frame + 1))
    overall_spot_score = 0.0
    hyp_intersects = False
    for _word, l, r, score in word_alignment:
        word_interval = set(range(l, r + 1))
        intersection_part = 100 / len(word_interval) * len(hyp_interval & word_interval)
        if intersection_part:
            if not hyp_intersects:
                overall_spot_score = score
            else:
                overall_spot_score += intersection_part / 100 * score
            hyp_intersects = True
    return overall_spot_score


def unbiased_headroom(
    context_graph: ContextGraphCTC,
    logprobs: np.ndarray,
    asr_model,
    blank_idx: int,
    ctc_ali_token_weight: float = 0.5,
) -> dict[str, float]:
    """One chunk's headroom score per term in context_graph that was
    spotted at all. A term absent from the result had no surviving
    candidate span in this chunk -- the caller (which sees every chunk
    in a clip) should treat a term missing from every chunk's result the
    same way: no acoustic trace found anywhere in the clip.
    """
    hyps = raw_spot(logprobs, context_graph, blank_idx, cb_weight=0.0)
    alignment = get_ctc_word_alignment(logprobs, asr_model, token_weight=ctc_ali_token_weight, blank_idx=blank_idx)

    scores: dict[str, float] = {}
    for hyp in hyps:
        gap = hyp.score - argmax_span_score(alignment, hyp.start_frame, hyp.end_frame)
        if hyp.word not in scores or gap > scores[hyp.word]:
            scores[hyp.word] = gap
    return scores
```

Note the parameter order on `unbiased_headroom` in the implementation
(`context_graph, logprobs, ...`) matches the test calls above, which use
a mix of positional and keyword args — keep them consistent with this
signature when writing Task 3.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `venv/bin/python -m pytest tests/test_headroom.py -v`
Expected: 7 passed.

- [ ] **Step 5: Run the full suite**

Run: `venv/bin/python -m pytest -q`
Expected: 147 passed, 1 skipped (140 before this task, +7 here).

- [ ] **Step 6: Commit**

```bash
git add captionlm/headroom.py tests/test_headroom.py
git commit -m "Add headroom.py: unbiased span-confidence signal for Task 4 validation"
```

---

### Task 2: capture-logits hook in `captionlm/biased_model.py`

**Files:**
- Modify: `captionlm/biased_model.py`
- Modify: `tests/test_biased_model.py`
- Modify: `docs/HANDOFF.md`

**Interfaces:**
- Consumes: nothing new — this only adds an attribute and three lines to the existing `generate()` method.
- Produces: `BiasedParakeetTDTCTC.capture_logits: list[np.ndarray] | None`, default `None`. When set to a list before a `transcribe()`/`generate()` call, that list is appended to (one array per internal chunk) as a side effect of the existing computation; behavior and output are otherwise unchanged. Task 3 sets this to `[]` and reads it back after calling `transcribe()`.

- [ ] **Step 1: Write the failing test**

Open `tests/test_biased_model.py` and add, after the existing three tests:

```python
def test_capture_logits_collects_one_array_per_chunk(tmp_path):
    wav_path = str(tmp_path / "silence.wav")
    _write_silent_wav(wav_path, seconds=1.0)

    model = load_biased_model(MODEL_ID)
    tokenizer = load_tokenizer(MODEL_ID)
    blank_idx = len(model.vocabulary)
    model.context_graph = build_context_graph(["kubernetes"], tokenizer, blank_idx)

    assert model.capture_logits is None  # default, before this change existed at all

    model.capture_logits = []
    model.transcribe(wav_path)

    assert len(model.capture_logits) >= 1
    assert all(isinstance(arr, np.ndarray) for arr in model.capture_logits)
    assert model.capture_logits[0].shape[1] == blank_idx + 1  # [T, vocab+blank]
```

Add `import numpy as np` at the top of the file alongside the existing imports.

- [ ] **Step 2: Run the test and confirm it fails**

Run: `venv/bin/python -m pytest tests/test_biased_model.py::test_capture_logits_collects_one_array_per_chunk -v`
Expected: FAIL — `AttributeError: 'BiasedParakeetTDTCTC' object has no attribute 'capture_logits'`.

- [ ] **Step 3: Add the hook**

In `captionlm/biased_model.py`, in `__init__` (currently lines 23-26):

```python
    def __init__(self, args: ParakeetTDTCTCArgs):
        super().__init__(args)
        self.context_graph: ContextGraphCTC | None = None
        self.spotter_config = SpotterConfig()
        self.capture_logits: list[np.ndarray] | None = None
```

In `generate()`, inside the existing per-batch-item loop (currently):

```python
            hypotheses = []
            for batch_idx, tokens in enumerate(result):
                logprobs = np.array(ctc_logits[batch_idx].astype(mx.float32))
                ws_hyps = run_word_spotter(
```

add the hook right after the `logprobs = ...` line:

```python
            hypotheses = []
            for batch_idx, tokens in enumerate(result):
                logprobs = np.array(ctc_logits[batch_idx].astype(mx.float32))
                if self.capture_logits is not None:
                    self.capture_logits.append(logprobs.copy())
                ws_hyps = run_word_spotter(
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `venv/bin/python -m pytest tests/test_biased_model.py -v`
Expected: 4 passed (the 3 existing GPU tests plus this one).

- [ ] **Step 5: Update the HANDOFF.md failure-count note**

In `docs/HANDOFF.md`, find:

```
The MLX smoke test is not paranoia. On the origin machine the Metal compiler's
XPC service died mid-session and every MLX call started failing with
`Compiler encountered XPC_ERROR_CONNECTION_INVALID`. The symptom is exactly
three failures in `tests/test_biased_model.py` while the other tests pass,
because those are the only tests that touch the GPU. It killed a running sweep
and it is fixed by a reboot, not by editing code. If you see those three
failures, stop and reboot -- do not debug them. (One `test_doc_import.py`
skip is normal on a machine without that fixture file; that's unrelated.)
```

Replace with:

```
The MLX smoke test is not paranoia. On the origin machine the Metal compiler's
XPC service died mid-session and every MLX call started failing with
`Compiler encountered XPC_ERROR_CONNECTION_INVALID`. The symptom is exactly
four failures in `tests/test_biased_model.py` while the other tests pass,
because those are the only tests that touch the GPU (a fourth joined the
original three on 2026-09-03, testing the headroom-signal capture hook). It
killed a running sweep and it is fixed by a reboot, not by editing code. If
you see those four failures, stop and reboot -- do not debug them. (One
`test_doc_import.py` skip is normal on a machine without that fixture file;
that's unrelated.)
```

Also update the `pytest -q` expected-count comment a few lines above it
(currently `# expect 140 passed (was 135 before the duration-cap tests)`)
to `# expect 141 passed (140 before this test)`.

- [ ] **Step 6: Run the full suite**

Run: `venv/bin/python -m pytest -q`
Expected: 148 passed, 1 skipped (147 after Task 1, +1 here).

- [ ] **Step 7: Commit**

```bash
git add captionlm/biased_model.py tests/test_biased_model.py docs/HANDOFF.md
git commit -m "Add opt-in capture_logits hook to BiasedParakeetTDTCTC for headroom-signal validation"
```

---

### Task 3: `scripts/measure_headroom_signal.py` — run it against real data

**Files:**
- Create: `scripts/measure_headroom_signal.py`

**Interfaces:**
- Consumes: `load_clips` from `captionlm.eval`; `per_term_stats` from `captionlm.eval`; `load_tokenizer`, `build_context_graph` from `captionlm.terms`; `load_biased_model` from `captionlm.biased_model`; `unbiased_headroom` from `captionlm.headroom`; `MODEL_ID` from `captionlm.config`.
- Produces: no importable interface — this is a terminal script, verified by running it, matching `scripts/sweep_cb_weight.py`'s own precedent (no test file for that script either).

- [ ] **Step 1: Write the script**

Create `scripts/measure_headroom_signal.py`:

```python
"""Does the headroom signal in captionlm/headroom.py actually predict
whether biasing a term will help or hurt? Retrospective: reads a sweep
already run by scripts/sweep_cb_weight.py, captures the unbiased CTC
logits for the same clips (one extra transcription pass per clip -- the
logits were never persisted, so this is unavoidable), computes each
listed term's headroom score, and checks whether that score actually
separates terms the sweep recovered from terms it falsely injected.

See docs/superpowers/specs/2026-09-03-headroom-signal-validation-design.md.
Nothing here is wired into cli.py -- this is a measurement, not a feature.
"""
import argparse
import json
import os

from captionlm.biased_model import load_biased_model
from captionlm.config import MODEL_ID
from captionlm.eval import load_clips, per_term_stats
from captionlm.headroom import unbiased_headroom
from captionlm.terms import build_context_graph, load_tokenizer


def _load_sweep_predictions(sweep_dir: str, condition: str, weight: float) -> dict[str, dict]:
    """clip basename -> {"reference", "baseline_pred", "biased_pred"}, as
    already recorded by scripts/sweep_cb_weight.py's per-weight JSON."""
    path = os.path.join(sweep_dir, f"sweep-{condition}-{weight}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {row["clip"]: row for row in data["per_clip"]}


def _clip_headroom(model, tokenizer, clip: dict, blank_idx: int) -> dict[str, float]:
    """Max headroom score per term across every chunk of this one clip."""
    model.context_graph = build_context_graph(clip["terms"], tokenizer, blank_idx)
    model.capture_logits = []
    model.transcribe(clip["wav"], chunk_duration=120.0)

    combined: dict[str, float] = {}
    for chunk_logprobs in model.capture_logits:
        chunk_scores = unbiased_headroom(
            model.context_graph, chunk_logprobs, model, blank_idx,
            ctc_ali_token_weight=model.spotter_config.ctc_ali_token_weight,
        )
        for term, score in chunk_scores.items():
            if term not in combined or score > combined[term]:
                combined[term] = score
    return combined


def _outcome(row: dict) -> str:
    if row["biased_correct"] > row["baseline_correct"]:
        return "recovered"
    if row["biased_false_accepts"] > row["baseline_false_accepts"]:
        return "injected"
    return "neither"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip_dir", help="Directory of <clip>.wav/<clip>.txt/<clip>.terms.txt")
    parser.add_argument("--sweep-dir", required=True, help="Directory holding sweep-<condition>-<weight>.json")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    clips = [c for c in load_clips(args.clip_dir) if c["terms"]]
    if not clips:
        print(f"no clips with a .terms.txt found in {args.clip_dir}")
        return

    predictions = _load_sweep_predictions(args.sweep_dir, args.condition, args.weight)

    model = load_biased_model(args.model)
    tokenizer = load_tokenizer(args.model)
    blank_idx = len(model.vocabulary)

    rows: list[dict] = []
    for clip in clips:
        basename = os.path.basename(clip["wav"])
        if basename not in predictions:
            print(f"skip {basename}: no matching entry in {args.sweep_dir}'s sweep JSON")
            continue

        pred = predictions[basename]
        headroom = _clip_headroom(model, tokenizer, clip, blank_idx)

        baseline_sample = [{"text": pred["reference"], "pred_text": pred["baseline_pred"]}]
        biased_sample = [{"text": pred["reference"], "pred_text": pred["biased_pred"]}]
        term_rows = per_term_stats(clip["terms"], baseline_sample, biased_sample)

        for row in term_rows:
            rows.append(
                {
                    "clip": basename,
                    "term": row["term"],
                    "outcome": _outcome(row),
                    "headroom": headroom.get(row["term"]),
                }
            )

    by_outcome: dict[str, list[float]] = {"recovered": [], "injected": [], "neither": []}
    no_candidate = {"recovered": 0, "injected": 0, "neither": 0}
    for row in rows:
        if row["headroom"] is None:
            no_candidate[row["outcome"]] += 1
        else:
            by_outcome[row["outcome"]].append(row["headroom"])

    print(f"\n{'outcome':<12} {'n (scored)':<12} {'n (no candidate)':<18} {'mean headroom':<15} {'min':<10} {'max':<10}")
    for outcome, scores in by_outcome.items():
        if scores:
            mean_s, min_s, max_s = sum(scores) / len(scores), min(scores), max(scores)
        else:
            mean_s = min_s = max_s = float("nan")
        print(f"{outcome:<12} {len(scores):<12} {no_candidate[outcome]:<18} {mean_s:<15.3f} {min_s:<10.3f} {max_s:<10.3f}")

    out_path = os.path.join(args.sweep_dir, f"headroom-{args.condition}-{args.weight}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nper-term rows written to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the real, corrected sweep**

The prerequisites first (per `docs/HANDOFF.md`, always before touching MLX):

```bash
venv/bin/python -m pytest -q
venv/bin/python -c "import mlx.core as mx; print(mx.arange(3))"
```

Then, on the four clips with real term lists in `data/eval_data/bias_clips/`,
against the corrected sweep from `2026-09-03-fixing-term-extraction-flips-per-clips-net-positive.md`:

```bash
nohup venv/bin/python scripts/measure_headroom_signal.py data/eval_data/bias_clips \
    --sweep-dir data/eval_results/sweep_earnings21_perclip_v2 \
    --condition per-clip --weight 2.0 \
    --model mlx-community/parakeet-tdt_ctc-1.1b \
    > /tmp/measure_headroom.log 2>&1 &
```

Budget similarly to one sweep cell on this clip set (the per-clip sweep's
own cb=2.0 pass took 1522s) — this pass does the same `transcribe()` call
as that sweep did, plus the cheap in-memory headroom computation on top.
Poll with `kill -0 <pid>` (not `pgrep` — see `docs/HANDOFF.md`'s traps).

- [ ] **Step 3: Read the result and decide what it means**

The printed table separates terms into `recovered` (the "Monro Forward"
family and anything else the sweep's biased pass got right that baseline
missed), `injected` (`Steve Winoker`, `Electric Distribution`, etc.), and
`neither`. The question this whole plan exists to answer: **does the mean
headroom score for `recovered` terms sit clearly above the mean for
`injected` terms?**

- If yes — the signal discriminates on real data. That's the result to
  write up in `docs/results/`, and the next step (not part of this plan)
  is deciding whether it's worth the two-pass cost as a production router.
- If no — same conclusion `self_recall` reached
  (`docs/results/2026-09-03-a-router-needs-a-better-headroom-proxy.md`):
  report it, and do not build a router on this proxy either.
- `no_candidate` counts matter too: a term the sweep recovered but that
  shows up with **no acoustic candidate at all** in the unbiased pass
  would be a real puzzle worth digging into before trusting the result
  either way — it would mean the signal missed exactly the case it's
  supposed to catch.

This step produces a finding, not a code change. Do not proceed to
writing a `docs/results/` doc or a follow-up plan without that finding in
hand — per `docs/HANDOFF.md`'s house rules, no number gets quoted before
it's actually been measured.

## Self-Review

**Spec coverage:** `raw_spot` (Task 1) ✓, `argmax_span_score` (Task 1) ✓,
`unbiased_headroom` (Task 1) ✓, `capture_logits` hook (Task 2) ✓,
`measure_headroom_signal.py` (Task 3) ✓, cross-reference via
`per_term_stats` (Task 3) ✓, HANDOFF.md failure-count update (Task 2) ✓.
Every component the spec named has a task.

**Placeholder scan:** no TBD/TODO; every step has real, verified code or
a concrete shell command.

**Type consistency:** `unbiased_headroom`'s signature
(`context_graph, logprobs, asr_model, blank_idx, ctc_ali_token_weight`)
is used identically in Task 1's tests and Task 3's `_clip_headroom` —
checked directly, not assumed, since Task 1 and Task 3 were written at
different points in this planning pass.
