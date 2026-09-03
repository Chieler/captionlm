# Handoff: continuing the B-WER work on another machine

Written 2026-09-02, updated 2026-09-03. Read
`docs/results/2026-09-02-what-b-wer-shows-that-wer-hides.md` first -- it is
the measurement record and this file only says what to do next. Tasks 1 and
2 below are now done (2026-09-03); their answer and the follow-on work it
opened are folded into this revision. If you're picking this up fresh, read
the results docs in dated order -- each one's "Limits" section is what the
next one exists to answer.

## Where things stand

`captionlm/eval.py:bias_wer` splits WER into B-WER (reference words on the
term list) and U-WER (everything else). It is wired into both arms of
`score()` and printed by `captionlm.eval` and `scripts/sweep_cb_weight.py`.

Settled:

- **The spotter works on genuinely rare vocabulary.** Read-aloud,
  `parakeet-tdt_ctc-1.1b`, per-clip term lists: B-WER 0.1343 -> 0.0763 at
  cb_weight 3.0, U-WER 0.0788 -> 0.0799. LibriSpeech rare-word benchmark
  (`is21_deep_bias`, 100 `test-clean` utterances): WER 0.0183 -> 0.0154 at
  cb_weight **2.0** (not 3.0 -- the first domain where NeMo's stock default
  isn't the answer), U-WER flat, injections at noise floor. See
  `docs/results/2026-09-02-librispeech-rare-words-is-the-case-that-works.md`.
- **It does not beat doing nothing on spontaneous speech with familiar
  vocabulary, in any configuration tried, but the story is not uniformly
  "pure cost."** Distractor list and oracle list (zero poisoned entries)
  are both net-negative at every `cb_weight` >= 1.0 tested -- a method
  problem on shared, cross-clip lists, not a list-quality problem, see
  Task 1 below. Per-clip (the actual shipped configuration) was also
  measured net-negative with zero terms ever recovered on 2026-09-02 --
  but that run turned out to be confounded by a term-extraction bug; once
  fixed (2026-09-03), the same 4 clips recover 17 term occurrences and
  net_term_gain turns positive (+13/+9 at cb=2.0/3.0), while
  whole-transcript WER still doesn't beat baseline. See Task 2 below. The
  dividing line for whether biasing has anything to fix is whether the
  baseline model already knows the words: it doesn't on LibriSpeech's
  rare-word set or on genuinely rare company-specific terms like "Monro
  Forward," and it does on Earnings-21's ordinary financial jargon.
- **cb_weight 3.0 is a real optimum on content the model doesn't already
  know**, bracketed on both sides -- the curve degrades below 2.0 and
  above 4.0, and at 8.0 U-WER nearly doubles. This optimum does not
  transfer to long-form spontaneous speech; see the duration cap below.

## Task 1 (DONE): oracle against distractor

**Answer: the method, not the list.** The oracle list (zero unspoken
padding, every entry genuinely said somewhere in the 11-clip set) is
net-negative at every weight tested, same as distractor. Precision
collapses (0.82 to 0.67, cb 1.0 to 3.0) while recall barely moves (0.71 to
0.77) -- a boost buying occasional recovered terms by rewriting ordinary
words it shouldn't touch. Two distinct false-accept mechanisms found by
reading the injected-term lists: short junk tokens (`LOS`, `TIM`, `GC`)
and legitimate terms from the wrong clip (`BOEING`, `CREDIT SUISSE` firing
on a call that never mentions them, because a shared list boosts one
clip's vocabulary against every clip in the batch). Full data and the
11-vs-9-clip count correction: `docs/results/2026-09-02-oracle-list-regresses-too.md`.

Consequence: **Task 3 (common-word guard) stays parked.** It was
conditioned on Task 1 blaming the list; it doesn't. A frequency filter
would not touch either false-accept mechanism found above -- `BOEING` is
not a common word, it's the wrong clip's word.

## Task 2 (DONE, then corrected 2026-09-03): the shipped path on spontaneous speech

**Original answer (2026-09-02): also net-negative.** Per-clip (each of 4
clips biased with only its own 50-term list -- the actual `cli.py`
pipeline, not a shortcut) removes cross-clip contamination entirely, and
the regression persisted anyway: `terms_recovered` was **0** at every
weight tested. The injected terms were generic financial-boilerplate
phrases (`levels`, `more information`, `share for the second quarter`).
Full data: `docs/results/2026-09-02-per-clip-also-regresses.md`.

**This diagnosis was itself confounded by a pipeline bug, found and fixed
2026-09-03.** `captionlm/build_eval_set.py` -- the script that built those
4 clips' `.terms.txt` files -- called the raw, unfiltered
`captionlm.term_extraction.extract_terms` instead of
`captionlm.biasable.extract_bias_terms`, the filtered extractor every
real product-facing script (`scripts/extract_terms.py`,
`scripts/caption_dropoff.py`, `scripts/serve.py`) already used. That raw
extractor is exactly what produces boilerplate phrases like "share for
the second quarter" -- `biasable.py`'s own docstring names this failure
mode. Fixed (`build_eval_set.py` now calls `extract_bias_terms(...,
mode="combined")`), term lists regenerated from the same cached filing
text (no new SEC fetches), sweep re-run:
**`terms_recovered` moved from 0 to 17, net_term_gain from -4/-9 to
+13/+9 at cb=2.0/3.0.** Whole-transcript WER still does not beat the
unbiased baseline at either weight -- that headline finding survives --
but "pure cost, zero recovery" does not. The recovered terms are almost
entirely one family ("Monro Forward", a program name the baseline got
wrong 14/14 times unbiased, right 13/14 times biased). Real false
accepts remain (6-10 per weight; `Steve Winoker`, `Electric Distribution`
-- genuinely rare terms, just not said in that clip). Full data and the
important caveat that this run's raw WER numbers aren't directly
comparable to the superseded doc's (different clip-pool size --
`net_term_gain` is the number that is comparable):
`docs/results/2026-09-03-fixing-term-extraction-flips-per-clips-net-positive.md`.

Only 4 of 11 real companies' filings made it through the SEC pipeline (7
failed on CIK lookup, missing 8-K, or a pyate `OverflowError` -- see
either per-clip doc for which). That gap is separate from the extraction
bug above and remains unfixed.

## What was tried as a fix, and what actually helped (2026-09-03)

Two `cb_weight`-selection strategies were benchmarked against each other:
static (pick the weight from list size and audio duration, before
transcription) vs. adaptive (run an unbiased pass first, back off the
weight if a proxy suggests baseline already knows the list). Full
writeup: `docs/results/2026-09-03-a-router-needs-a-better-headroom-proxy.md`.

**The adaptive approach doesn't work and shouldn't be revived as-is.** Its
proxy (`self_recall`: fraction of the term list found in the *unbiased*
transcript) never once picked a different weight than the static rule, in
any regime tested. Root cause is structural, not a tuning miss: on real
per-clip lists, 63% of extracted terms are never spoken in that clip at
all, so "term absent from the unbiased transcript" is dominated by "wasn't
said," not "was said but missed" -- the two things the proxy needs to tell
apart. Whole-transcript presence cannot separate them at any threshold.

**One static rule helped, for free, and is now shipped:** `cb_weight = 1.0`
for audio over 300 seconds, `select_cb_weight()` in
[captionlm/config.py](captionlm/config.py) (wired into
`caption_file()` in [captionlm/cli.py](captionlm/cli.py), which probes
duration via `ffprobe` through `get_audio_duration()` before deciding the
weight). This cut Earnings-21 per-clip's damage from net -9 (unconditional
cb_weight=3.0) to net -2 in the one sweep that measured it. It does not
close the gap -- WER 0.1776 vs. baseline's 0.1773, still a hair worse --
and it costs nothing: duration is free, known before transcription starts,
no spotter changes.

## Task 3: validate the duration cap (do this before trusting it further)

The 300s threshold and the 1.0 fallback weight come from **one sweep**,
run ad hoc in the same session that found them (raw output:
`router_earnings21.json`, scratchpad, not committed). Before leaning on
this in a product claim, it needs the same rigor as everything else in
this file:

- Re-run on **all 11 Earnings-21 clips** now that the SEC pipeline's 4/11
  yield is understood (`docs/results/2026-09-02-per-clip-also-regresses.md`
  lists the three fixable-vs-structural reasons the other 7 failed) --
  n=4 is not enough to trust net -2 as a stable number.
  `venv/bin/python -m pytest -q` and the model smoke test still apply
  before any of this (see below).
- Check the threshold itself, not just the one data point at 3.0/1.0. Is
  300s the right cutoff, or does the regression start earlier/later? Is
  1.0 the best fallback for long-form audio, or would 0.5 or unbiased
  entirely do better? `scripts/sweep_cb_weight.py` already supports
  sweeping weights; it does not yet support sweeping the duration
  threshold itself, so that harness needs a small extension first.
- Confirm it doesn't regress anything on the short-clip regimes
  (LibriSpeech, read-aloud) where cb_weight=3.0 was already known good --
  the cap should never fire there (both regimes are well under 300s), but
  that's an assumption, not something explicitly re-checked end-to-end
  through `cli.py` after this change.
- `tests/test_config.py` and the `test_get_audio_duration` case in
  `tests/test_cli.py` cover the unit-level logic (threshold boundary,
  ffprobe parsing) but not whether 300s/1.0 are the *right* numbers --
  that's an eval question, not a test-suite question.

## Task 4 (DONE, 2026-09-03): the headroom signal

**Built and measured, not just proposed.** `captionlm/headroom.py`
computes per-span decode confidence at the specific locations a listed
term might plausibly occur -- not whole-transcript presence, which is
what made the adaptive router above fail. For each term, it runs the CTC
word spotter's beam search unbiased (`cb_weight=0`) to find its best
acoustic match, then compares that against what the model's own
argmax-decoded path already believes is at that span (the same
comparison `filter_wb_hyps` already relies on in production, just moved
earlier in the pipeline). A small opt-in `capture_logits` hook on
`BiasedParakeetTDTCTC` (no-op unless set, nothing wired into `cli.py`)
gets the raw CTC logits back out, since `generate()` normally discards
them. `scripts/measure_headroom_signal.py` runs the whole thing against
already-swept clips and cross-references the result against each term's
real fate (recovered / injected / neither).

**Answer: the signal is real but not yet proven at this sample size.**
Recovered terms score clearly better (mean headroom -1.442) than
injected (-2.268) or neither (-2.521) -- a term biasing correctly
recovers is ~73-76% likely to outscore a term it falsely injects or a
term irrelevant to recall, by this measure. That does not reach
conventional significance (Mann-Whitney p≈0.063) at n=6 recovered
terms, clustered in only 2 of the 4 clips. It also is not merely a
term-length artifact -- headroom correlates strongly with word count
(r≈-0.79), but the effect survives when restricted to single-word terms
only. Full writeup, including a casing bug that first made this look
like total failure until it was caught and fixed:
`docs/results/2026-09-03-headroom-signal-shows-a-real-but-unproven-effect.md`.

**What's next, if this gets picked back up:** more data before more
engineering. The signal itself works as designed; what's missing is
enough recovered-term examples to know whether the effect holds up --
a second sweep cell, a second cb_weight, or LibriSpeech's rare-word
benchmark (cleaner genuine-vs-distractor labels than Earnings-21's
extracted-term lists) would all add power cheaply, before spending
effort on the actual router logic (per-term thresholding) this
measurement doesn't attempt to design. Do not build that router logic
on this signal until it clears significance on more data than one
sweep cell provides.

## Before you run anything

```bash
venv/bin/python -m pytest -q                          # expect 147 passed (146 before this test)
venv/bin/python -c "import mlx.core as mx; print(mx.arange(3))"
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

## Traps that cost time on the origin machine

- **`cb_weight` 0.5 is inert** on both eval sets -- WER, B-WER and U-WER all
  unchanged to four decimals. Do not spend a 25-minute pass on it.
- **Long runs die with the terminal.** Use
  `nohup ... > run.log 2>&1 &` and poll the log.
- **`pgrep` lies when `sysmond` is down** -- it exits non-zero with
  `Cannot get process list`, which a `while pgrep` watcher reads as "finished".
  Use `kill -0 <pid>` or `ps -p <pid>` instead.
- **The sweep's unbiased baseline is one shared ~21-minute pass** reused by
  every cell. Killing the sweep forfeits it; there is no resume.
- **Budget ~16 min per pass** for the Earnings-21 eval10 clips on an
  unloaded machine, roughly 25 under contention. A 2-condition x 3-weight
  grid is ~2 hours including the baseline. Earlier docs in this project
  (including the first version of this file) say "9 clips" --
  `eval10-file-metadata.csv` has always had **11** rows and nothing
  explains the discrepancy; budget from 11, not 9. See
  `docs/results/2026-09-02-oracle-list-regresses-too.md`.

## Do not

- **Do not quote the Earnings-21 distractor numbers as product numbers.**
  They come from a hostile list applied uniformly to every clip.
- **Do not read read-aloud's monotone improvement as proof a weight is safe.**
  Every row there reads `inject=n/a`: the document *is* the transcript, so
  there are no listed-but-unspoken terms and no denominator for hallucinating
  one. That set can detect over-biasing through U-WER but can never detect a
  wrong term list.
- **Do not restart the live/streaming work** without addressing its own gate.
  It failed GATE 2 at +5.8-6.5pp live-vs-batch, ~3x over threshold, and the
  cause is structural: commit-boundary deletions plus a 30s context window
  against batch's 120s. See the live-streaming plan under `docs/superpowers/plans/`.
- **Do not claim the duration cap "fixes" long-form biasing.** It is a
  stopgap validated by one n=4 sweep, shipped because it's free and
  strictly better than the unconditional default -- not because it closes
  the gap. It doesn't (0.1776 vs. baseline's 0.1773). Task 3 above is
  re-validating it; until that's done, treat net -2 as directional, not a
  number to quote.
- **Do not claim the headroom signal "works" or build a router on it yet.**
  It shows a real, direction-consistent effect that survives a length-
  confound check, but at n=6 recovered terms it does not reach
  significance (p≈0.06-0.10) and those 6 cluster in only 2 clips. The
  first run said the opposite (a term-casing bug zeroed out exactly the
  class that mattered) -- treat any future extension of this signal with
  the same suspicion until it, too, survives independent re-derivation
  from raw output. See
  `docs/results/2026-09-03-headroom-signal-shows-a-real-but-unproven-effect.md`.

## Getting the data

`SETUP.md` has the full sequence. Short version: read-aloud `.wav` and
`.terms.txt` are tracked, so that set runs straight after install.
Earnings-21 is 1.7 GB and is not in git -- clone
`revdotcom/speech-datasets` into `data/speech-datasets/` and run
`scripts/build_bias_clips.py`.
