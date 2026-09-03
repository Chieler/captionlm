# The headroom signal shows a real but statistically unproven effect -- and a casing bug nearly said the opposite

`parakeet-tdt_ctc-1.1b`, full-clip batch decode (`chunk_duration=120s`),
Earnings-21 per-clip (the 4 clips with real SEC-filing term lists:
Monro, GE, Pilgrim's Pride, Eversource), `cb_weight=2.0`, validated
against `data/eval_results/sweep_earnings21_perclip_v2/` -- the corrected
sweep from `2026-09-03-fixing-term-extraction-flips-per-clips-net-positive.md`,
not the superseded buggy-extractor run. This is Task 4 from
`docs/HANDOFF.md`: does the model's own unbiased decode confidence at a
term's span predict whether biasing it would help or hurt?

Full design: `docs/superpowers/specs/2026-09-03-headroom-signal-validation-design.md`.
Implementation: `captionlm/headroom.py`, `scripts/measure_headroom_signal.py`,
a small opt-in `capture_logits` hook on `BiasedParakeetTDTCTC` -- nothing
wired into `cli.py`, this is a measurement, not a feature.

## The method, briefly

For each term in a clip's bias list, run the CTC word spotter's beam
search unbiased (`cb_weight=0`, no boost) to find its best acoustic
match, then compare that against what the model's own argmax-decoded
path already believes is at that span (`get_ctc_word_alignment`, the
same comparison `filter_wb_hyps` already uses in production to filter
spotted hypotheses). The gap between the two -- `headroom` -- should be
large when the model is genuinely uncertain there (worth boosting) and
small or negative when the model already has a confident answer
(boosting is close to pure risk).

## A casing bug nearly produced the opposite conclusion

The first run said the signal was completely dead: 0 of 6 genuinely-
recovered terms (the ones biasing actually fixed, per the sweep) scored
*any* candidate at all -- not a low score, no score, full stop. That
looked like damning evidence the whole approach was unsound. It wasn't
real. `scripts/measure_headroom_signal.py`'s cross-reference step
compared a dict keyed by each term's *original casing* (from the term
list file, via `raw_spot`) against `per_term_stats`' rows, which are
*always lowercased* (`vendor/fscore.py`'s `keyword_stats` does this
unconditionally). Every one of the 6 recovered terms (`Store`, `Monro`,
`Acquisition`, `Capital`, `ACTIONS`, `GE Capital`) happened to be
capitalized in its source file, so the lookup could never succeed --
independent of what the spotter actually found. Fixed with a one-line
`.lower()` normalization; the measurement was re-run clean, not patched.
Two independent reviews (one on the bug, one re-verifying the fix by
recomputing every statistic from the raw output JSON from scratch)
confirmed the mechanism and the fix. The lesson: a join between two
differently-cased term representations can silently zero out exactly
the class of result that matters most, with no error and a plausible-
looking result on the other side.

## Results, after the fix

| outcome | n scored | n no-candidate | mean headroom | min | max |
|---|---|---|---|---|---|
| recovered | 6 | 0 | -1.442 | -2.883 | -0.602 |
| injected | 14 | 1 | -2.268 | -4.115 | -0.998 |
| neither | 60 | 12 | -2.521 | -7.807 | -0.820 |

Recovered terms score clearly better (less negative) than both other
buckets. A rank-based effect size: a randomly picked recovered term
outscores a randomly picked injected term ~73% of the time (61/84
pairs), and a `neither` term ~76% of the time (275/360 pairs). That is
a real, moderate, hypothesis-consistent effect.

**It does not reach conventional statistical significance at this
sample size.** Mann-Whitney U (recovered vs. injected, n=6 vs n=14)
gives p≈0.063; Welch's t-test gives p≈0.10. Six recovered terms is not
enough to confirm a hypothesis, only to not yet reject it.

Per-term detail, recovered bucket (all 6 scored, sorted best to worst):

```
store          -0.602   (Monro, clip 4320211)
actions        -0.930   (GE, clip 4341191)
capital        -1.045   (GE, clip 4341191)
monro          -1.195   (Monro, clip 4320211)
acquisition    -2.000   (Monro, clip 4320211)
ge capital     -2.883   (GE, clip 4341191)
```

## Two things that qualify the effect further

**Effective sample size is closer to 2 than 6.** All 6 recovered terms
come from only 2 of the 4 clips (Monro's 3, GE's 3), and within each
clip they are nested fragments of one entity, not independent draws
(`capital` and `ge capital` are scored off overlapping acoustic spans by
construction). Treating them as 6 independent observations for the
significance test above already overstates the evidence; the honest
frame is closer to "2 clips showed the pattern," which is suggestive,
not close to sufficient.

**Headroom correlates strongly with term length (r ≈ -0.79 across all
80 scored terms), and the buckets aren't length-matched** -- recovered
is 5/6 single-word, injected is 10/15 single-word. A signal that mostly
just measures "how many tokens is this" would trivially look like it
discriminates outcome, if outcome itself correlates with length (which
it plausibly does: short, common-sounding words are both easier to
recover by biasing and easier for a spotter to find *some* acoustic
match for). Restricting to single-word terms only tests this directly:
recovered mean -1.154 (n=5) vs. injected mean -1.664 (n=10), pairwise
concordance 0.72 -- essentially unchanged from the pooled 0.73. **The
effect survives controlling for length.** It is not purely an artifact
of longer terms scoring worse across the board.

## What this does and doesn't establish

**Does:** on this one sweep cell, headroom shows a real, moderate,
direction-consistent separation between terms biasing actually helped
on and terms it falsely injected, and that separation is not just term
length in disguise. The mechanism (comparing unbiased spotted evidence
against the argmax path's own confidence at the same span) measures
something real, not noise.

**Doesn't:** prove the signal is strong enough to build a router on. n=6
recovered terms, clustered in 2 clips, does not clear a significance
threshold. Nothing here has been checked against a second sweep cell,
a second cb_weight, or a different eval set (LibriSpeech's rare-word
benchmark would be a natural next check, given it has much cleaner
genuine-vs-distractor labels). And a working headroom-based router would
still need per-term thresholding logic this measurement doesn't attempt
to design -- knowing the signal correlates with outcome is not the same
as knowing where to draw the line.

## Limits

Four clips, one condition, one weight -- the same small-n caveat every
per-clip Earnings-21 measurement in this project carries
(`docs/results/2026-09-03-fixing-term-extraction-flips-per-clips-net-positive.md`
already flagged n=4 clips as underpowered; this analysis is a further
subdivision of that same small set, down to the term level). The
casing bug this doc opens with is a reminder that this whole
measurement pipeline is new and its first real run was wrong in a way
that looked exactly like a clean negative result -- treat any future
extension of this method with the same suspicion until it, too, survives
an independent re-derivation of its numbers from raw output.

Full per-term data: `data/eval_results/sweep_earnings21_perclip_v2/headroom-per-clip-2.0.json`
(gitignored, not committed -- regenerate with `scripts/measure_headroom_signal.py`
against the same sweep directory to reproduce).
