# What contextual biasing can and cannot fix

Measured 2026-08-31 on the read-aloud eval set (`read_aloud/`), two
recorded technical scripts, 1517 reference words. Every number below is
reproducible from `scripts/sweep_cb_weight.py`; the raw JSON lands in
`eval_results/` and is git-ignored.

## Why this set exists

Earlier measurement on Earnings-21 could not answer whether biasing
works, because it confounded three failures: a miscalibrated spotter, a
bad term list, and the method not working. It also ran into a hard data
ceiling — across three earnings calls, 47 ASR errors were fixable in
principle and only 5 appeared anywhere in the company's own press
release. Eleven percent. No extractor can beat that.

The read-aloud set removes the confound by construction: the script is
simultaneously the source document the extractor reads and the reference
transcript the WER is scored against. A failure here is attributable to
the extractor or the spotter, never to the data source.

That makes it an **upper bound, not a product estimate**. A real user's
document is related to their audio, not identical to it.

## Headline

| model | unbiased | biased, cb=3.0 | relative |
|---|---|---|---|
| parakeet-tdt_ctc-110m | 0.1160 | 0.0989 | -14.7% |
| parakeet-tdt_ctc-1.1b | 0.1009 | **0.0778** | -22.9% |

End to end, 110m unbiased to 1.1b biased: 0.1160 -> 0.0778, **-33%
relative**. Term F-score over the same span: 0.8610 -> 0.9464.

Per clip, 1.1b: ai_agents 0.0908 -> 0.0697, distributed_systems
0.1110 -> 0.0859.

## cb_weight is 3.0, and the earlier reading was wrong

Measured on 110m. Clear inverted U:

```
0.25  0.1160   (bit-identical to baseline; the spotter does not fire)
1.0   0.1167
2.0   0.1042
3.0   0.0989   <- optimum
4.0   0.0989
6.0   0.1285
8.0   0.2037
12.0  0.4726
```

3.0 is the default `SpotterConfig` already ships. Weights at and above
6.0 were measured against an earlier, fragment-heavier term list; the
shape is unambiguous but those three cells are not directly comparable to
the rest.

An earlier sweep over Earnings-21 suggested the useful region was *below*
0.5. That was an artifact: it scored a 1013-term corpus list against
clips containing almost none of it. With a term list drawn from the
audio's own document, the stock weight is right.

## The error decomposition, which is the real finding

Error spans from `jiwer` alignment, at cb_weight 3.0:

| model | baseline | biased | fixed | introduced | residual on terms | residual elsewhere |
|---|---|---|---|---|---|---|
| 110m | 144 | 122 | 59 | 37 | 35 | 87 |
| 1.1b | 125 | 96 | 45 | 16 | 33 | 63 |

Three things follow.

**Most of the residual is not a biasing problem.** 63 of 96 remaining
errors at 1.1b are on words that are not domain terms: `the` deleted, `a`
deleted, `often` deleted, plus hallucinated insertions (`quad`, `agents`,
`been`). At 110m, 63 of those 87 were single words of four characters or
fewer. No term list reaches them. This is acoustic model capacity.

**Biasing and model size are complementary, not substitutes.** Biasing
gains *more* on the larger model (-0.0231 vs -0.0171), and residual
term errors barely move between the two (35 -> 33). The big model does
not fix terms; biasing does not fix grammar.

**The spotter damages its neighbours, and the cause is `merge_spans`.**
It introduces 37 new errors at 110m and 16 at 1.1b while fixing 59 and
45 — a 26% collateral rate even at the better model.

Decoding is not the culprit and never was. `BiasedParakeetTDTCTC.generate`
already runs an unbiased TDT decode, spots over the CTC head separately,
and splices the two with `merge_spans`; the acoustic search is never
perturbed. Instrumenting `merge_spans` over both clips at cb_weight 3.0
shows the real mechanism: 16 replacements are lossy, destroying 16 words
outright, and they share one shape.

```
'the hard parts'      -> 'hard parts'
'the base weights'    -> 'base weights'
'a small model'       -> 'small model'
'the wrong tool.'     -> 'wrong tool'
'a fencing token,'    -> 'fencing token'
'protocol to'         -> 'protocols'
```

A spotted keyword whose frame span overlaps the preceding article by the
`intersection_threshold` of 30% consumes that article, and
`merge_spans` replaces the whole consumed range with the keyword alone.
That is exactly the deleted-`the`, deleted-`a` pattern in the introduced
errors. The replacement histogram shows the same thing from the other
side: 152 clean 2-word-for-2-word replacements against 14 that turn
three words into two.

**Correction to an earlier reading in this document's first revision.**
The duplicated span `"Grock built custom silicon Grock built custom
silicon"` was attributed here to a defect in the merge logic. It is not.
It is present in the *unbiased* baseline at both model sizes; the
acoustic model repeats the phrase on its own, and biasing only shifts
where the aligner charges the error. Nothing in the spotter or the merge
produces it.

## What biasing structurally cannot fix

**Homophones.** `Groq` and `Grok` are *both* in the term list for
ai_agents and are acoustically identical; the spotter has no mechanism to
choose. Same for `write`/`right`, `writes`/`rights`, `weights`/`waits`. A
context graph matches audio to token sequences and has no access to
sentence-level context, so no term list and no weight can resolve these.

**Morphology.** `subtrees` -> `subtree`, `objects` -> `object`,
`shorter` -> `short`, `monotonic reads` -> `monotonistic read`. Partial
matches against an uninflected list entry. This one is cheaply fixable by
adding inflected variants.

**Terms that should never have been in the list.** `better`, `real`,
`lets` and `traces` entered extraction as noise and now cause
substitutions of their own.

## Term list size: measured, but confounded

At cb_weight 3.0 on 110m, WER improved monotonically with list size all
the way to the full extraction (10 terms: 0.1167; 50: 0.1055; 100:
0.1015; 200: 0.0995; full: 0.0936).

`TERM_EXTRACTION_TOP_N` was **not** raised on this evidence. The gain is
contaminated by the upper-bound artifact: a fragment such as
`"dense open"` is literally spoken here because the document *is* the
transcript, and is pure noise on any real document. Removing such
fragments from the extractor raised measured WER from 0.0936 to 0.0989,
which is the correct direction. The cb_weight result does not share this
contamination, because the wrong-document arm corroborates it
independently.

## Wrong-document arm

Biasing a clip with the *other* script's terms cost +0.0066, against
-0.0250 for the clip that got its own. The right document helps roughly
four times more than the wrong one hurts.

## Cost

| | 110m | 1.1b |
|---|---|---|
| on disk | 438 MB | 4.0 GB |
| 11.3 min of audio | 11 s | 24 s |
| realtime factor | ~47x | ~28x |

Neither involves training at use time, so both satisfy the project's
central constraint.

## Conclusion

Biasing is a **term-accuracy** tool, not a WER tool. It drives term
F-score from 0.86 to 0.95 and buys roughly 20% relative WER. It cannot
fix grammar, function words, or homophones. If overall WER is the goal,
the remaining budget is in model choice and in second-pass rescoring
that has access to sentence-level context.
