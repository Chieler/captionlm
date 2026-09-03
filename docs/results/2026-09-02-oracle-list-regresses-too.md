# The oracle list regresses too -- Task 1's diagnosis is the method, not the list

`parakeet-tdt_ctc-1.1b`, full-clip batch decode (`chunk_duration=120s`, not
the parked streaming path), Earnings-21 eval10 -- **11 clips**, not the 9
prior docs in this project cite. `eval10-file-metadata.csv` has always had
11 rows; nothing in the pipeline explains the earlier "9," and no per-clip
breakdown survives to say which 2 an earlier run dropped. Treat this run's
numbers as the correction, not as incomparable to the 9-clip figures below
-- direction agrees regardless of the 2-clip difference.

## What this separates

The distractor-list run recorded in
`2026-09-02-what-b-wer-shows-that-wer-hides.md` was net-negative: B-WER
flat, U-WER worse, injections ten times the chance floor. That result
can't say *why* -- the distractor list is the oracle list plus ~769
Fortune 500 names and CEOs that are never spoken, so a regression there is
consistent with either "biasing is fragile on spontaneous speech" or "the
list is poisoned." Running the oracle list alone -- zero unspoken padding,
every entry genuinely said somewhere in these calls -- at the same
weights, scored against the same (distractor) vocabulary, separates the
two.

## Results

| condition | cb_weight | B-WER | U-WER | WER | net terms | injection rate |
|---|---|---|---|---|---|---|
| unbiased baseline | -- | 0.1348 | 0.2011 | 0.1835 | -- | -- |
| oracle | 1.0 | 0.1352 | 0.2022 | 0.1844 | -29 | 0.0048 (51/10701) |
| oracle | 2.0 | 0.1349 | 0.2027 | 0.1847 | -37 | 0.0072 (77/10701) |
| oracle | 3.0 | 0.1347 | 0.2036 | 0.1853 | -55 | 0.0124 (133/10701) |
| distractor | 1.0 | 0.1352 | 0.2028 | 0.1848 | -36 | 0.0032 (61/19139) |
| distractor | 2.0 | 0.1349 | 0.2034 | 0.1852 | -52 | 0.0050 (95/19139) |
| distractor | 3.0 | 0.1349 | 0.2045 | 0.1860 | -85 | 0.0085 (163/19139) |

No swept point in either condition beat the unbiased baseline's WER. The
lowest WER in the whole sweep is oracle at cb_weight 1.0 (0.1844), still
0.0010 worse than doing nothing.

## Diagnosis: the method, not the list

The oracle arm is net-negative at every weight tested, and it contains
**no poisoned entries at all**. B-WER never meaningfully improves over
baseline -- 0.1348 to a best case of 0.1347, one word in ten thousand.
U-WER gets steadily worse as weight increases (0.2011 to 0.2036).
Precision collapses (0.82 to 0.67 across cb 1.0 to 3.0) while recall
barely moves (0.71 to 0.77) -- the classic shape of a boost that is
buying occasional recovered terms by rewriting ordinary words it
shouldn't touch.

Per `docs/HANDOFF.md`'s decision rule, this is the first branch: **biasing
itself is fragile on spontaneous speech; the method needs work, not the
list.** The distractor arm is worse than oracle at every matched weight
(net -85 vs -55 at cb=3.0), so the poisoned entries do add real damage on
top -- but they are not the root cause. A perfectly clean list still
loses to doing nothing.

The injected-term lists back this up in a way the aggregate numbers
don't show on their own. Most repeat offenders are short junk tokens that
fire at every weight regardless of condition -- `LOS`, `BI`, `EV`, `TIM`,
`GC`, `IP`, `JAN`, `GL`, `DS`, `CHAR`, `COVER`, `PV`, `SELECT`, `NG`,
`UD`, `ASA`, `DT`, `CI` -- the same failure mode already named in the
distractor run (`pay`, `jan`, `operations`: common English words that
happen to sit on a term list). But some are unambiguous, non-generic
entities that are still false accepts on clips that never say them --
`BOEING`, `CREDIT SUISSE`, `INVESTOR RELATIONS`, `CAPEX`,
`PRIVATE SECURITIES LITIGATION REFORM ACT`. These are real, meaningful,
correctly-spelled financial terms. They are on the oracle list because
some *other* clip's call genuinely says them. The context graph doesn't
know which clip it's biasing; it boosts the whole oracle vocabulary
against every one of the 11 calls, and a big enough list of legitimate
terms is enough on its own to manufacture false accepts on audio that
never mentions them. A common-word frequency filter (Task 3) would not
touch this class of injection -- `BOEING` and `CREDIT SUISSE` are not
common words, they are simply the wrong clip's words.

## Consequence for the gate

Task 3 (a common-word guard in `captionlm/terms.py`) was explicitly
conditioned in `docs/HANDOFF.md` on Task 1 blaming the list: *"Do not
build it before Task 1 says the list is the problem."* It doesn't. The
oracle arm has zero junk entries and still regresses, and a meaningful
share of its false accepts are legitimate terms misapplied to the wrong
clip rather than generically common words. Building a frequency filter
would not fix either failure mode. That task stays parked.

## Not a streaming artifact

Both arms ran through `captionlm/eval.py`'s `transcribe_baseline` /
`transcribe_biased`, which call `model.transcribe(wav, chunk_duration=120.0)`
-- the full-clip batch path, with the whole call available as context.
This rules out the live/streaming path's known failure mode (30s context
window, commit-boundary deletions, +5.8-6.5pp live-vs-batch gap) as an
explanation. The fragility measured here is a property of context-graph
biasing itself, independent of and separate from the streaming work that
remains parked under its own gate.

## Limits

Three weights (1.0, 2.0, 3.0); 0.5 is already known inert on both eval
sets and was skipped. No per-clip breakdown was pulled from the sweep
JSONs -- whether the false accepts concentrate on a handful of clips or
spread evenly is unmeasured. Whether a *per-clip* term list (Task 2, the
shipped configuration) shows the same fragility, or whether restricting
each clip's context graph to only its own document's terms removes the
cross-clip false-accept mode described above, is the next open question.

Full per-weight data: `data/eval_results/sweep_earnings21/sweep-summary.json`.
