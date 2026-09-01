# What it takes to reach ~5% WER

Measured on the read-aloud set, `parakeet-tdt_ctc-1.1b`, biased with each
clip's own extracted term list at `cb_weight` 3.0 — the configuration
`cli.py` and `scripts/caption_dropoff.py` actually ship.

**Current: 119 errors over 1517 reference words, WER 0.0784.**
**Target: 76 errors. 43 error words have to go — 36% of the residual.**

> **Superseded in two places, 2026-09-01.** Items 1 and 2 have both been
> measured. Item 1 shipped at −0.0073. Item 2 — document-conditioned
> rescoring — was tried four ways and every one of them made the transcript
> worse; what reached the same failures was a second acoustic model rather
> than a language model, at −0.0152, taking the set to 85 errors and 0.0560.
> That work also found that **44 of these 119 error words are not ASR errors
> at all** — Whisper makes them identically, because the reader departed from
> the script — which falsifies the inflection row below. See
> `2026-09-01-a-second-opinion-on-the-same-audio.md`. The buckets and the
> "what will not work" section stand; the per-item forecast does not.

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
| inflection only, stem correct | 5 | `protocol` → `protocols`, `objects` → `object` (**not ASR errors — the reader said these**) |
| homophone | 4 | `groq` → `grok` (**both listed**), `write` → `right` |

## What each change is worth

| # | change | reaches | words | ΔWER | confidence |
|---|---|---|---|---|---|
| 1 | disfluency suppression | reader restarts | ~~14~~ **11** | ~~−0.0092~~ **−0.0073** | ~~high~~ **done** |
| 2 | document-conditioned rescoring | inflection, homophones, function, ordinary content | ~30 of 61 | ~~−0.020~~ | ~~medium~~ **falsified; a second acoustic model got −0.0152 instead** |
| 3 | per-term `cb_weight` | garbled listed terms | 3–4 of 7 | −0.002 | medium-low |
| 4 | remaining merge collateral | some function words | 2–4 | −0.002 | low |

**1 + 2 + 3 + 4 → 0.049.** Reaches the target.
**Item 1 is now measured and shipped: 0.0784 → 0.0712.**
**Everything except 2 → 0.065.** Does not.

Rescoring is load-bearing. Every other change on this list, combined, is
worth about 0.013, and there is no fifth item hiding: the buckets above
are exhaustive over the residual.

### 1. Disfluency suppression — MEASURED: −0.0073, shipped

**Result: WER 0.0784 → 0.0712, 119 → 108 errors.** All eleven come out
of insertions (35 → 24); substitutions and deletions are untouched, so
the pass is strictly subtractive. Zero false positives over the
1517-word read-aloud reference set. See `captionlm/disfluency.py`.

The budget below predicted 14 words. The shippable detector gets 11, and
the gap is instructive: the estimate was made with a detector that asks
which inserted words also appear in the *reference* nearby, and there is
no reference at inference time. What ships instead keys on the property
that makes a restart a restart — an n-gram immediately followed by
itself — which finds the repeats but not the three words of
`'' → 'always part is'`, where the reader restarted with different words.

One caveat that did not exist before measuring: this is correct for a
caption and wrong for a verbatim transcript. Earnings-21 references
contain the speakers' real disfluencies, and the same pass removes 391
of their 153,088 reference words. It is applied in `result_to_srt`, the
product path, and not in eval scoring against verbatim references.

#### Original estimate

Ten spans, all pure insertions whose words repeat reference text within
a twelve-word window: `'' → 'the failure modes'`, `'' → 'always part is'`,
`'' → 'applications'`. The reader restarted mid-sentence and the model
transcribed it faithfully. Deterministic to detect and, for a caption
product, correct to drop — a caption that shows the speaker's false
starts is worse than one that does not.

Two-for-one: it is a real product feature and it removes measurement
noise that inflates every WER number this project has recorded.

### 2. Document-conditioned rescoring — MEASURED: it does not work

**Four variants, all worse than doing nothing** (120, 104, 104 and 118
errors against 108). A second acoustic model reached the same buckets for
−0.0152. The reasoning below is kept because the buckets it names are still
the right ones; the conclusion it draws from them is not.

The failures it reaches are exactly the ones no term list can:

- `groq` → `grok`. Now measured end to end in
  `2026-09-01-what-the-trie-can-and-cannot-encode.md`: both are spotted
  on identical frames, `grok` wins both by 2.898 and 4.084, and the CTC
  head's own argmax is `ck` in both places. No acoustic or trie-side
  lever reaches it, and — correcting this document's earlier claim — the
  *document* does not disambiguate them either, because it contains both
  ("the name is a constant source of confusion with Grok"). What
  separates them is the surrounding sentence: "___ built custom silicon
  for low latency decoding" against "confusion with ___". That is
  available only once a transcript exists, which is what makes this
  rescoring's job and nothing else's.
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

The spike now has an exact target to hit. Region 0 of `ai_agents.wav`,
frames 1202–1205, currently reads "Grok built custom silicon" and should
read "Groq built custom silicon"; region 1, frames 1347–1351, reads
"Grok" and is already correct. A rescoring pass has to flip the first
without touching the second — which is precisely the discrimination a
single global term weight cannot make, and a sentence-conditioned pass
can.

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
