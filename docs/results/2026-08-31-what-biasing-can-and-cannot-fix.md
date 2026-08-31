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

**Correction, twice over.** The duplicated span `"Grock built custom
silicon Grock built custom silicon"` was first attributed to a defect in
the merge logic, then to the acoustic model repeating itself. Both are
wrong. The reader restarted that sentence while recording, and the model
transcribed the restart faithfully. The tell was already visible: it
appears identically at both model sizes, which a hallucination would not
do. Nothing in the spotter, the merge, or the model produces it.

## Disfluency: a known confound in the reference

The reader stuttered and restarted several times. The reference is the
script, so every disfluency the model transcribes correctly is scored as
an insertion. Of 22 insertions in the unbiased 1.1b hypotheses, six carry
a clear disfluency signature:

```
'the chunking strategy is [always part is] almost always the part'   restart
'the failure [models the failure] modes'                             restart
'the wrong tool they [fall they] fail'                               stutter
'into a sequential [attempt] appends'                                partial word
'the trained [mod] weights'                                          partial word
'fragmented grock built custom silicon [grock built ...]'            restart
```

The others are genuine model errors that happen to insert: `[bar] rails`
for "guardrails", `[nt] entropy` for "anti-entropy", `[quark] code` for
"Claude".

Six of 96 residual error spans at 1.1b is roughly 6%, so **absolute WER
on this set is inflated by something on the order of 0.005**. Quote the
numbers here as comparisons, not as accuracy.

Every comparison in this document survives intact, because all of them
run the same audio through different configurations: the cb_weight
optimum, the biased-versus-unbiased delta, the 110m-versus-1.1b delta and
the merge instrumentation are all differences over a shared reference,
and a constant inflation cancels.

What it does change is one attribution above: part of the "residual
elsewhere" bucket credited to acoustic model capacity is the model
getting the audio *right* against a reference that does not match it.

**Removing disfluency does not fit the current design.** The spotter
matches audio to token sequences and `merge_spans` splices frame spans;
neither can know a speaker restarted. Collapsing "always part is almost
always the part" needs sentence-level understanding, which is the same
machinery as the deferred document-conditioned rescoring pass. It is
folded into that spike rather than planned separately.

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

## After the 2026-08-31 precision plan

Four tasks were coded against the findings above. Two landed, two were
reverted by controller ruling after being measured against the plan's own
0.002 WER tolerance. This section reports what actually reached the
tree, not what the plan hoped for.

| model | before | after | fixed | introduced (before) | introduced (after) |
|---|---|---|---|---|---|
| 110m, cb=3.0 | 0.0989 | 0.0969 | 56 | 37 | 31 |
| 1.1b, cb=3.0 | 0.0778 | 0.0784 | 45 | 16 | 17 |

Term-list sizes are unchanged from the plan's starting point: 206
(ai_agents), 242 (distributed_systems). Both tasks that would have
changed them (Task 3's plural variants, Task 4's NOUN/PROPN restriction)
were reverted, so there is nothing here for term-list size to explain.

**110m moved as the plan predicted.** WER 0.0989 -> 0.0969, introduced
errors 37 -> 31, both below the Step 2 expectation of "below 37." This is
Task 1 alone; Task 2 has no WER effect by design.

**1.1b did not move as predicted, and the honest reading is that it got
slightly worse.** WER 0.0778 -> 0.0784 (+0.0006), and introduced errors
went 16 -> 17 — one *more*, not fewer, against the Step 2 expectation of
"below 16." Fixed count held at 45. This is a small movement, on the
order of one span out of 96, and could be run-to-run noise in the merge
boundary rather than a systematic regression, but the plan's own rule is
that a number moving the wrong way gets written down, not rationalized
away. It is not being treated as a reason to revisit Task 1: Task 1's
mechanism (not consuming more words than the spotted keyword accounts
for) is directionally correct and the 110m result confirms it works:
1.1b apparently has one edge case where the trim interacts differently
with an already-more-confident spotter. Worth a look if a future plan
touches `merge_spans` again; not worth reopening this one for.

**Per task, what it was worth:**

- **Task 1** (`merge_spans` stops over-consuming words at the low-overlap
  edge): the only change with a measured WER effect. Landed. Worth 0.0020
  WER at 110m (0.0989 -> 0.0969) and, on the same measurement, a
  small net cost at 1.1b (+0.0006). Net positive across both sizes but
  not uniformly so.
- **Task 2** (name the two TDT-CTC checkpoints; raise `ValueError` on a
  `blank_idx` / tokenizer mismatch): landed, zero WER effect by design.
  It closes a silent-failure hole rather than changing any output on
  passing inputs, so no WER movement is the expected and correct result,
  not a null finding.
- **Task 3** (match regular plural surface variants in the trie):
  **reverted.** Measured WER rose to 0.0995 (+0.0026 over baseline,
  outside the plan's 0.002 tolerance), and recall *fell* 0.9053 ->
  0.8915 on both clips — the opposite of what adding matches should do,
  which means the plural entries were displacing real hits rather than
  adding to them. Worth nothing as shipped. One fragment survived: a
  shared-prefix trie insertion-order fix (feed shorter variants first,
  since `ContextGraphCTC.add_to_graph` reused a shared-prefix node
  without updating `is_end`, silently making the base term
  un-spottable), which is a correctness fix independent of the plural
  feature and is why it was kept while the rest was reverted.
- **Task 4** (restrict rare-lowercase term harvest to `NOUN`/`PROPN`):
  **reverted.** Its premise — that `better`, `lets`, `real`, `traces` are
  "ordinary English the model already transcribes correctly" — was
  tested and is false: aligning the unbiased baseline against the
  reference shows those four words spoken 7 times combined, misrecognized
  4 of those 7 times (`better` 1/1, `lets` 1/2, `real` 1/3, `traces`
  1/1). They carry real recall upside that a POS filter would have thrown
  away; measured WER rose to 0.0995 (+0.0026, same magnitude as Task 3
  and for a related reason: removing them narrows the term list without
  removing genuine error sources). Worth nothing as shipped; worth
  falsifying its own premise, which is the point of measuring before
  shipping a spec's assumptions.

**Net of this plan:** Task 1 plus Task 2. 110m WER improved by 0.0020;
1.1b WER moved by -0.0006 (i.e., got worse by that amount) under the
same change, both at cb_weight 3.0 with per-clip terms. Two of four
coded tasks were reverted, and the revert record itself is the more
durable output of Task 3 and Task 4: both hypotheses (plurals help
recall; POS-filtering removes only noise) were specific, testable, and
false, which is worth knowing for the next plan even though neither
line of code shipped.
