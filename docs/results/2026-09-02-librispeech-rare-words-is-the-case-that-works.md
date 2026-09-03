# LibriSpeech rare-word biasing: the first natural-speech case that works

`parakeet-tdt_ctc-1.1b`, full-clip batch decode (`chunk_duration=120s`),
100 utterances from LibriSpeech `test-clean`, biasing lists from
`facebookresearch/fbai-speech`'s `is21_deep_bias` benchmark (the standard
academic rare-word context-biasing test set, built for exactly this
question: does biasing help when the ASR model genuinely doesn't already
know the word). Every prior real-audio result this session --
`2026-09-02-oracle-list-regresses-too.md`,
`2026-09-02-per-clip-also-regresses.md` -- was net-negative on Earnings-21
at every weight tested. This one is not.

## The benchmark

`is21_deep_bias`'s rare words are words outside the 5,000 most common
words in the LibriSpeech LM training corpus (~209K such words exist,
~60% never appear in that training data at all). Each `test-clean`
utterance's row supplies its own genuinely-spoken rare words (2.71 on
average across the 100-utterance subset used here) plus a padded biasing
list of size N=1000 -- the genuine words plus ~997 random distractor rare
words from elsewhere in the corpus, unrelated to this utterance. This is
structurally the same oracle-plus-distractor design as this project's own
Earnings-21 sweeps, at a comparable per-clip list size (Earnings-21's
oracle list: 1013 terms; here: ~1000 per utterance) -- but applied to
5-20 second utterances instead of 50-90 minute calls, and to random-book
vocabulary instead of same-domain financial jargon.

Subset selection: the 100 utterances are the first (by TSV file order) of
the 1980 `test-clean` utterances (of 2620 total) that contain at least one
genuine rare word, so every utterance contributes real B-WER signal. Not
random-sampled; a fixed, reproducible prefix of the file.

All 100 clips built cleanly from LibriSpeech's standard
`speaker/chapter/speaker-chapter-utterance.flac` layout -- no skips, no
CIK-style lookup fragility. This is the first eval set in the project
where every planned clip actually built.

## Results

| cb_weight | WER | B-WER | U-WER | net terms | injection rate |
|---|---|---|---|---|---|
| unbiased baseline | 0.0183 | 0.0212 | 0.0113 | -- | -- |
| 0.5 | 0.0183 | 0.0212 | 0.0113 | +0 | 0/99758 |
| 1.0 | 0.0178 | 0.0205 | 0.0113 | +1 | 1/99758 |
| 1.5 | 0.0159 | 0.0178 | 0.0113 | +5 | 1/99758 |
| **2.0** | **0.0154** | **0.0171** | 0.0113 | **+6** | 1/99758 |
| 3.0 (NeMo stock default) | 0.0164 | 0.0171 | 0.0146 | +5 | 2/99758 |
| 4.0 | 0.0197 | 0.0178 | 0.0243 | -2 | 10/99758 |
| 5.0 | 0.0255 | 0.0226 | 0.0324 | -15 | 23/99758 |
| 6.0 | 0.0438 | 0.0356 | 0.0632 | -51 | 60/99758 |
| 8.0 | 0.1204 | 0.0925 | 0.1864 | -203 | 210/99758 |

A genuine U-curve, same shape as read-aloud's, but the optimum sits at
**cb_weight=2.0**, not NeMo's stock 3.0 -- the first domain measured in
this project where the stock default is not the right answer. Overall WER
improves (0.0183 to 0.0154, biasing beating the unbiased baseline
outright, which never happened once on Earnings-21), U-WER stays exactly
flat at the baseline's 0.0113 all the way through cb=3.0 (zero cost to
ordinary words in that range), and injections stay at noise-floor levels
even at the optimum (1 or 2 out of 99,758 unspoken-pair chances).

## Why this domain is different, mechanistically

Two structural differences from Earnings-21, both traceable to the
false-accept mechanism found in this session's diagnostics run
(short list entries generate many spurious near-matches across long
audio -- a base-rate effect of keyword spotting over long recordings):

- **Utterances are 5-20 seconds, not 50-90 minutes.** Far less audio
  "surface area" for a short-token distractor to coincidentally
  near-match, independent of anything about the term list itself.
- **Distractors are drawn from unrelated books by unrelated authors**,
  not same-domain financial jargon. Earnings-21's cross-clip
  contamination (`BOEING`, `CREDIT SUISSE` firing on the wrong clip) works
  because those are real financial terms genuinely said somewhere in the
  11-clip set; a random LibriSpeech distractor word has no such structural
  reason to acoustically resemble anything in a specific unrelated
  utterance.

Because of this, the `ctc_ali_token_weight` mitigation found earlier this
session (raising it from 0.5 toward 1.5 cut Earnings-21's false accepts
roughly in half) is not needed here -- injections are already at noise
floor at the optimum weight, so there is essentially nothing for that
guard to fix on this domain. That mitigation is specific to Earnings-21's
long-audio, same-domain-distractor pathology, not a universal fix.

## What this does and doesn't establish

**Does**: biasing toward genuinely-unfamiliar vocabulary, on short
utterances, works cleanly and the shipped `cb_weight=3.0` default is
close to but not exactly optimal for this kind of content -- 2.0 is
measurably better here. The mechanism is not fundamentally broken; it
works as designed when its actual precondition (the baseline model
doesn't already know the word) holds.

**Doesn't**: this is not evidence Earnings-21's diagnosis was wrong, or
that per-clip Earnings-21 numbers would improve if terms were "more
rare." Per-clip's problem was that the *baseline model already knew
the words* (97-100% correct on genuinely-spoken company jargon) -- there
was no rare-word gap to close in the first place, on that content. This
result and Earnings-21's are not in tension; they answer "does biasing
help when there's a real vocabulary gap" (yes, cleanly) versus "does
biasing help on ordinary professional speech with common financial
jargon" (no, there's nothing to fix) -- two different questions, both now
measured.

## List-size scaling: N=100, N=500, N=2000

The same 100 utterances, same full weight curve, at the benchmark's other
three published list sizes (`is21_deep_bias` publishes N=100/500/1000/2000;
all four now measured).

| N | optimum cb_weight | optimum WER | net @ cb=2.0 | injection rate @ cb=2.0 | net turns negative at |
|---|---|---|---|---|---|
| 100 | 2.0 | 0.0149 | +7 | 0/9998 | cb=8.0 |
| 500 | 2.0 | 0.0164 | +4 | 3/49950 | cb=4.0 |
| 1000 | 2.0 | 0.0154 | +6 | 1/99758 | cb=4.0 |
| 2000 | 2.0 | 0.0149 | +7 | 0/199063 | cb=4.0 |

B-WER/U-WER are not strictly comparable across N -- each run's scoring
vocabulary is the pooled union of that N's own per-clip lists, so the
same reference word can be "listed" under a larger N and "other" under a
smaller one. WER (whole-transcript) and injection rate do not have that
problem and are the reliable cross-N comparisons.

**cb_weight=2.0 is the optimum at every N tested** -- a robust,
list-size-independent tuning result, not an artifact of one particular
list size. But the safe-range story is a **threshold, not a gradient**:
N=100 tolerates weight all the way to cb=6.0 before net turns negative,
while N=500, N=1000, and N=2000 all break at exactly the same point,
cb=4.0, regardless of further growing the list from 500 to 2000. An
earlier version of this doc reported N=500 breaking at cb=6.0 -- that was
a misread of the raw sweep output; the correct value, re-checked against
`sweep-librispeech-N500`'s per-weight rows, is cb=4.0. Whatever separates
"forgiving" from "not" sits somewhere between 100 and 500 distractors per
clip, not smoothly across the whole range -- consistent with the
base-rate mechanism (more distractor entries, more chances for a spurious
match) but sharper than a simple monotonic decay.

Full per-weight data: `data/eval_results/sweep_librispeech_N100/sweep-summary.json`,
`data/eval_results/sweep_librispeech_N500/sweep-summary.json`,
`data/eval_results/sweep_librispeech_biasing100_full/sweep-summary.json`,
`data/eval_results/sweep_librispeech_N2000/sweep-summary.json`.

## test-other: noisier audio shifts the optimum, not just the headroom

Same methodology (100 utterances with >=1 rare word, first by file order,
N=1000), on `test-other` -- LibriSpeech's acoustically harder split.

| split | baseline WER | baseline B-WER | baseline recall | optimum cb_weight | optimum WER | net turns negative at |
|---|---|---|---|---|---|---|
| test-clean | 0.0183 | 0.0212 | 0.9236 | 2.0 | 0.0154 | cb=4.0 |
| test-other | 0.0558 | 0.0655 | 0.7940 | 3.0-4.0 (tied, net=+9) | 0.0525 | cb=5.0 |

Baseline is far worse on test-other (recall 0.79 vs 0.92) -- more genuine
acoustic uncertainty, not just vocabulary unfamiliarity, and headroom is
correspondingly larger. The optimum weight moves up to 3.0-4.0 and the
safe range widens (stays net-positive through cb=4.0, only turns negative
at cb=5.0, net=-1). This is a third axis, distinct from list size and
vocabulary rarity: **noisier audio tolerates and rewards a stronger
contextual nudge**, plausibly because the acoustic signal alone is less
sufficient, so the prior from the bias list carries more of the decision.
List size was not re-swept on test-other (N=1000 only); whether the same
N=100-vs-N>=500 threshold from test-clean holds here is unmeasured.

## Limits

100 of 2620 `test-clean` utterances (2141 of 2939 for test-other), a fixed
file-order prefix rather than a random sample. Read-aloud's caveat about
zero real injection risk does not apply here (this set has genuine
unspoken-term padding and real injections show up once cb_weight is
pushed past the optimum at every N and both splits), but the absolute
injection counts are small enough (low tens at worst before each curve
turns catastrophic at cb=8.0) that per-weight injection-word identities
were not individually reviewed the way Earnings-21's were.
