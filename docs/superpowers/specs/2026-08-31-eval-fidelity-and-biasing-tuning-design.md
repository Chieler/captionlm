# Eval Fidelity and Biasing Tuning — Design

Date: 2026-08-31
Status: draft, pending implementation plan

## Problem

The glue-and-eval pipeline ran for the first time on 2026-08-30 and
produced a negative result: context biasing made things worse.

| run | WER | P | R | F |
|---|---|---|---|---|
| baseline (2 clips, post-`5d49542`) | 0.260 | 0.938 | 0.979 | 0.958 |
| biased | 0.262 | 0.849 | 0.982 | 0.910 |

Recall moved +0.003. Precision fell −0.089. WER rose +0.002.

The shape of that result — recall flat, precision down — is the signature
of a spotter that is accepting hypotheses it should reject, not one that
is finding terms the baseline missed. Before concluding that CTC-WS
biasing does not help this domain, four defects in the *measurement* have
to be removed, because each one independently pushes the numbers in
exactly the direction observed.

A follow-up review found ten issues. Five were mechanical and are already
fixed on `master` (jiwer normalization, context-graph case variants,
`extract_call_year` range guard, exception typing, `rmtree` guard).

The jiwer fix needed two corrections found while writing this plan:
`jiwer.wer` takes `reference_transform`, not `truth_transform` (the first
attempt would have raised `TypeError` at runtime, uncaught because no test
exercised it), and `jiwer.wer_standardize` is itself unusable here — it
includes `RemoveWhiteSpace`, which glues a sentence into a single token.
`eval.py` now composes `ToLowerCase / RemovePunctuation /
RemoveMultipleSpaces / Strip / ReduceToListOfListOfWords` explicitly, with
a regression test.

The five remaining issues need structural change and are the subject of
this spec.

## Central Constraint

**The user must never wait for a model to train.**

This constraint is not at risk. The user-facing runtime path is
`captionlm/cli.py:31-41` in full:

```python
model = load_biased_model(model_id)          # load pretrained MLX weights
if term_list_path:
    tokenizer = load_tokenizer(model_id)
    terms = load_term_list(term_list_path)
    model.context_graph = build_context_graph(terms, tokenizer, blank_idx)
result = model.transcribe(audio_path, chunk_duration=120.0)
```

`build_context_graph` is a trie insertion over the term list — O(total
subword tokens), milliseconds for a 50-term list. Nothing in this path
fits, trains, or adapts weights. That is the entire point of CTC-WS
context biasing over fine-tuning: the domain knowledge lives in a data
structure built at call time, not in the weights.

Every change in this spec lands in one of two places, neither of which is
the user's runtime path:

1. **The eval harness** (`eval.py`, `build_eval_set.py`) — developer-side,
   never shipped, never invoked by `cli.py`.
2. **A constant in `config.py`** — the `cb_weight` sweep is a one-time
   offline calibration run by a developer. The user inherits the tuned
   number as a default. The user never sweeps.

**Verdict: none of the five structural changes violate the constraint.**

The only user-visible first-run latency is a download, not a train: the
~110M-parameter model pulled from HuggingFace and cached by
`hf_hub_download`. Subsequent runs hit the cache.

## Changes

### S1 — Per-clip term lists

`build_eval_set.py:262` unions every clip's extracted terms into one
`terms.txt`, and `eval.py:43` loads that single file for all clips. With
`TERM_EXTRACTION_TOP_N = 50` and 5 clips, the context graph carries ~250
terms, of which ~200 belong to other companies' filings.

**Why this raises WER.** Every term in the graph is a spot candidate at
every frame. A term drawn from another company's filing is, by
construction, not spoken in this clip — it can *only* produce a false
accept. Each false accept that survives `filter_wb_hyps` reaches
`merge.py:merge_spans`, which replaces the overlapping transducer word
span with the spotted term. That is a substitution error the baseline did
not have, and often an insertion/deletion pair when token counts differ.
Irrelevant terms therefore contribute a strictly one-directional push:
WER up, precision down, recall unchanged.

That is precisely the observed signature.

**Why this is a correctness fix, not a tuning knob.** `cli.py` takes one
`--terms` file for one audio file. The product biases a document's terms
toward *that document's* call. The current eval measures a configuration
the product never runs. Fixing it makes the experiment measure the
shipped thing.

**Expected effect:** graph shrinks ~5×. Most of the −0.089 precision gap
should close. WER delta should move from +0.002 toward negative.

### S2 — `cb_weight` calibration sweep

`config.py:20` ships NeMo's stock `cb_weight = 3.0`. Its own docstring
says to retune before trusting it. Neither run did.

**Why this raises WER.** In `vendor/ctc_word_spotter.py:326`:

```python
current_score = token.score + logprobs[frame][int(transition_state)].item() + cb_weight
```

`cb_weight` is added **once per non-blank token transition**, so the bonus
accumulates with term length. `'net sales'` tokenizes to 5 pieces, so it
collects up to **+15.0** of pure bonus.

Two gates are supposed to reject bad spots, and the accumulated bonus
defeats both:

1. `ctc_word_spotter.py:340` emits a hypothesis when
   `new_token.score > keyword_threshold` (−5.0). A 5-token term carrying
   +15.0 needs a total acoustic logprob worse than −20.0 to fail — roughly
   −4.0 per token, which is very poor acoustic evidence but far from
   impossible on noisy earnings-call audio.
2. `filter_wb_hyps` (`ctc_word_spotter.py:218-248`) is the actual
   false-accept guard: it compares the spot's score against the CTC
   alignment score for the overlapping frames and drops the spot if the
   alignment scores higher. **The spot's score includes the accumulated
   `cb_weight`; the CTC alignment's does not.** At `cb_weight = 3.0` the
   comparison is rigged by +3 per token in the keyword's favor, so the
   guard is systematically defeated — and defeated *hardest* for long
   multi-word terms, which is exactly what domain term extraction produces.

This is the direct mechanical explanation for precision −0.089 with recall
+0.003: the spotter is not finding new terms, it is failing to reject
wrong ones.

**Expected effect:** sweeping `cb_weight` down should restore
`filter_wb_hyps` to working order. There is a point below 3.0 where true
spots still clear the guard and false ones stop doing so. WER should fall
below baseline there. If no such point exists across the sweep, *that* is
the honest negative result — and unlike the current one, it will be a
claim about the method rather than about an untuned constant.

**User cost: zero.** The sweep is offline and developer-side. The output
is one float in `config.py`.

### S3 — Persist per-clip eval output

`run_eval` builds a rich result and `main()` prints four floats
(`eval.py:88-90`). Nothing is written to disk. The 2026-08-30 artifacts
lived in a gitignored directory inside a deleted worktree, so a ~4-hour
run survives only as the numbers in this document.

This is not cosmetic — it is what makes S2 affordable. A 5-point
`cb_weight` sweep as currently structured re-transcribes the baseline arm
five times, and the baseline does not depend on `cb_weight` at all. Half
the compute in a sweep is redundant by construction.

**Why this improves WER:** it does not, directly. It makes the S2 sweep
cheap enough to actually run, and makes any WER number auditable
per-clip instead of a single opaque average across a 2-clip sample.

### S4 — Prefer EX-99.1 over other EX-99.x exhibits

`build_eval_set.py:136-139` returns the first `index.json` entry matching
`ex99`/`ex-99`, in directory order, with no preference for `.1`.

**Why this raises WER.** EX-99.2 on an earnings 8-K is typically the slide
deck or supplemental financial tables. Running `pyate` over a table dump
yields column headers and row labels — `"Three Months Ended"`,
`"Diluted EPS"`, fragmentary line items — text that is *formatted*, not
spoken. Those terms enter the context graph but are never uttered on the
call, so like S1's cross-clip terms they can only ever false-accept.

EX-99.1 is the press release: continuous prose that executives read from
and paraphrase aloud. Terms extracted from it are actually spoken.

**Expected effect:** a higher share of graph terms become capable of true
accepts, and the dead weight that can only false-accept drops. Recall up,
false-accept pressure down, WER down.

### S5 — EDGAR `filings.recent` pagination

`build_eval_set.py:109` reads only `filings.recent`, which
`submissions.json` caps at ~1000 entries. Older filings live in
`filings.files[]` and are never fetched. Earnings-21 covers 2020–2021, and
for a high-volume filer 1000 filings is well under two years.

**Why this matters for WER:** it does not change per-clip WER. It changes
the sample. Two clips survived assembly; a WER delta of 0.002 on n=2 is
indistinguishable from noise, and the earlier 5-clip run is not comparable
(different clips, different denominator). Paginating should lift the
assembled set toward 15–20 clips.

**Honest framing:** this improves *confidence* in the WER number, not the
number. It is the difference between "biasing hurt" and "biasing hurt,
measurably".

### S6 — Delete dead lookup code, correct the spec

`eval_dataset.resolve_cik`, `find_press_release_filing`, and
`assemble_triple` have zero production callers — only their own tests.
`build_eval_set.find_earnings_release_filing` is a near-copy of
`find_press_release_filing` plus the Item 2.02 filter.

The 2026-08-30 design spec also still asserts that "jiwer's default
transform lowercases and strips punctuation on both sides anyway". That is
false for jiwer 4.0.0 (`wer_default` is
`[RemoveMultipleSpaces, Strip, ReduceToListOfListOfWords]`), and the
Task-2 review cited it as the "cost if wrong is zero" justification for
parking a real finding. The spec also describes `build_eval_set` importing
`find_press_release_filing`, which `5d49542` made stale.

**Why this improves WER:** it does not. Zero runtime impact. It is here
because the spec currently documents a false premise that was already used
once to dismiss a genuine bug, and will be again by the next reader.

## Summary

| # | Change | Touches user runtime? | Improves WER? |
|---|---|---|---|
| S1 | Per-clip term lists | No — eval harness | Yes, directly |
| S2 | `cb_weight` sweep | No — offline constant | Yes, largest lever |
| S3 | Persist eval output | No — eval harness | No; enables S2 |
| S4 | Prefer EX-99.1 | No — eval-set builder | Yes, indirectly |
| S5 | EDGAR pagination | No — eval-set builder | No; improves confidence |
| S6 | Delete dead code | No | No; hygiene |

S1 and S2 are the two that should move the number. S4 supports them by
improving term quality; S3 and S5 make the result reproducible and
statistically meaningful. None of the six require training, and none add
latency to `cli.py`.

## Non-goals

- No fine-tuning, adapter training, or LM fusion. The no-training
  constraint is the architecture, not a limitation to work around.
- No change to `merge.py`'s `intersection_threshold` or to the vendored
  NeMo algorithms beyond parameters already exposed in `SpotterConfig`.
- No new runtime dependencies.
