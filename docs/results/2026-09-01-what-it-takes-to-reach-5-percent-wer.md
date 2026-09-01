# What it takes to reach ~5% WER

Measured on the read-aloud set, `parakeet-tdt_ctc-1.1b`, biased with each
clip's own extracted term list at `cb_weight` 3.0 — the configuration
`cli.py` and `scripts/caption_dropoff.py` actually ship.

**Current: 119 errors over 1517 reference words, WER 0.0784.**
**Target: 76 errors. 43 error words have to go — 36% of the residual.**

## The budget

Every residual error span, bucketed. Reproduce with the alignment in
`captionlm/eval.py:_NORMALIZE` over the stored predictions in
`eval_results/final_1b/sweep-product-3.0.json`.

| bucket | words | spans | WER share |
|---|---|---|---|
| touches a domain term | 53 | 40 | 0.0349 |
| ordinary content word | 33 | 28 | 0.0218 |
| function word (articles, prepositions) | 19 | 19 | 0.0125 |
| reader restart (disfluency) | 14 | 10 | 0.0092 |

**52 of the 53 term-bucket error words involve a term that was in that
clip's bias list.** Extraction is not the bottleneck. The extractor found
these terms, the trie held them, the spotter had every chance, and they
came out wrong anyway. Only one term error in the whole set
(`application` deleted) involves a term the extractor missed.

Splitting those 39 spans by what actually went wrong:

| failure | spans | example |
|---|---|---|
| acoustically distant substitution | ~23 | `quorum` → `core`, `vector clock` → `different machinery` |
| listed term garbled anyway | 7 | `linearizability` → `delarizability`, `anthropic` → `ntpic` |
| inflection only, stem correct | 5 | `protocol` → `protocols`, `objects` → `object` |
| homophone | 4 | `groq` → `grok` (**both listed**), `write` → `right` |

## What each change is worth

| # | change | reaches | words | ΔWER | confidence |
|---|---|---|---|---|---|
| 1 | disfluency suppression | reader restarts | 14 | −0.0092 | high |
| 2 | document-conditioned rescoring | inflection, homophones, function, ordinary content | ~30 of 61 | −0.020 | medium |
| 3 | per-term `cb_weight` | garbled listed terms | 3–4 of 7 | −0.002 | medium-low |
| 4 | remaining merge collateral | some function words | 2–4 | −0.002 | low |

**1 + 2 + 3 + 4 → 0.047.** Reaches the target.
**Everything except 2 → 0.065.** Does not.

Rescoring is load-bearing. Every other change on this list, combined, is
worth about 0.013, and there is no fifth item hiding: the buckets above
are exhaustive over the residual.

### 1. Disfluency suppression — 0.0092, cheapest thing on the list

Ten spans, all pure insertions whose words repeat reference text within
a twelve-word window: `'' → 'the failure modes'`, `'' → 'always part is'`,
`'' → 'applications'`. The reader restarted mid-sentence and the model
transcribed it faithfully. Deterministic to detect and, for a caption
product, correct to drop — a caption that shows the speaker's false
starts is worse than one that does not.

Two-for-one: it is a real product feature and it removes measurement
noise that inflates every WER number this project has recorded.

### 2. Document-conditioned rescoring — the only lever big enough

The failures it reaches are exactly the ones no term list can:

- `groq` → `grok`. Both terms are in the list. They are acoustically
  identical. No acoustic method distinguishes them; only the document
  says which one this speaker meant.
- `protocol` → `protocols`. Word-final /s/ under a following consonant.
  Adding plural variants to the trie was tried and reverted (it made
  things worse); the number is a language-model decision, not an
  acoustic one.
- 33 words of ordinary content and 19 of function words. Not domain
  vocabulary, so no term list ever reaches them.

The constraint is unchanged and absolute: **no training, ever.** This is
a prompted pass over the finished transcript plus the source document,
not adaptation.

Open questions the spike must answer before anything ships: does it fix
`Grok` → `Groq` without introducing new errors, and at what latency.
It could plausibly return worse text than it was given.

### 3. Per-term `cb_weight` — small, cheap, now safely measurable

Seven spans are listed terms the spotter simply lost:
`linearizability` → `delarizability`, `replica` → `bobpeka`,
`antientropy` → `entropy`. A long rare term needs more push than a short
common one, and today every term gets the same weight.

The global weight is exhausted — 3.0 is the floor of the curve, and 4.0
buys recall (0.9231 → 0.9270) while costing WER (0.0784 → 0.0798).
Per-term weighting is the only remaining move in that direction, and
`biasing_yield` plus the injection rate now make it safe to try: they
say immediately whether extra push is recovering terms or hallucinating
them.

## What will not work

**A bigger acoustic model.** Already spent: 110m → 1.1b was −0.021, the
largest single win recorded. `mlx-community` publishes no larger
TDT-CTC parakeet. Switching model family breaks CTC-WS outright — the
spotter needs a CTC head over the same sentencepiece vocabulary, which
is why this architecture was chosen.

**More or better term extraction.** 52 of 53 term errors are on terms
already in the list. A better extractor recovers at most one word.

**Raising `cb_weight` globally.** Measured, inverted-U, 3.0 is optimal;
past 6.0 it collapses.

**Pruning junk terms.** Real on raw lists — 79 of 178 Earnings-21 false
accepts come from 19 terms that never once produced a true positive —
but `is_biasable` already rejects 11 of those 19, killing 63 of the 79.
On our own extracted lists there is exactly one such term
(`entropy`, one false accept).

## The caveat that governs all of it

This budget is the read-aloud set: one speaker, clean audio, scripted
prose, and a document that IS the transcript. Real material sits far
higher — Earnings-21 is at 0.19 WER, and 5% there is not reachable by
anything on this page. "~5% WER" is a claim about clean single-speaker
audio with a matching source document, and should be quoted that way.
