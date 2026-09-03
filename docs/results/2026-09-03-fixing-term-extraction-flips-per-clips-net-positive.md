# Fixing the term-extraction bug flips per-clip's net term gain from negative to positive

`parakeet-tdt_ctc-1.1b`, full-clip batch decode (`chunk_duration=120s`),
Earnings-21 `data/eval_data/bias_clips/` (11 clips, 4 with real per-clip
term lists), scored against the distractor list's 1782-term vocabulary,
same methodology as `2026-09-02-per-clip-also-regresses.md`.

## What changed

`captionlm/build_eval_set.py:317` called `captionlm.term_extraction.extract_terms` --
pyate's raw statistical extractor, with no filtering -- to build each
clip's `.terms.txt`. `captionlm/biasable.py` already existed to fix
exactly that extractor's known failure mode (its own docstring names the
same pathology this project's own per-clip eval hit: generic
boilerplate phrases scoring high enough to make pyate's top-50 cut). The
real product-facing scripts (`scripts/extract_terms.py`,
`scripts/caption_dropoff.py`, `scripts/serve.py`) all already called the
filtered `captionlm.biasable.extract_bias_terms`; `build_eval_set.py`
never got updated to match. This was a pure oversight, not a deliberate
choice -- `scripts/extract_terms.py`'s own docstring says its `combined`
mode "is the mode the product would actually ship."

Fixed by changing `build_eval_set.py` to call
`extract_bias_terms(filing_text, mode="combined", top_n=TERM_EXTRACTION_TOP_N)`
instead. The four clips whose SEC pipeline succeeded
(`4320211`/Monro, `4341191`/GE, `4383161`/Pilgrim's Pride,
`4384964`/Eversource) had their `.terms.txt` regenerated from the
already-cached `.doc.txt` filing text -- no new SEC EDGAR fetches needed.
Before/after samples, same filing, same top-50 cutoff:

| clip | old (unfiltered) sample | new (filtered) sample |
|---|---|---|
| Monro | `+`, `long`, `lesser extent`, `levels` | `Monro`, `shares outstanding`, `Diluted`, `Diluted EPS` |
| GE | `investor`, `future`, `leverage goals` | `GE Capital`, `BioPharma`, `Industrial`, `NYSE` |
| Pilgrim's | `lead to product liability claims`, `obligations` | `Pilgrim`, `PRIDE CORPORATION`, `U.S.` |
| Eversource | `ability`, `loss`, `liabilities as a whole` | `Eversource Energy`, `EPS`, `NPT` |

A rough word-overlap check against each clip's own reference transcript
(not the eval's exact whole-phrase metric, just a sanity check) found
126/200 old terms had zero word overlap with what was actually said;
35/200 do under the new extractor. Both lists still share 17-26/50 terms
per clip -- legitimate rare domain words neither extractor's difference
touches -- so this is a reduction in junk, not a wholesale replacement.

## Results

Re-ran the same sweep (`scripts/sweep_cb_weight.py`, `--condition
per-clip=per-clip`, weights 2.0/3.0) against the regenerated term lists.

| condition | cb_weight | WER | B-WER | U-WER | precision | recall | F | terms recovered | injections | net |
|---|---|---|---|---|---|---|---|---|---|---|
| unbiased baseline | -- | 0.1835 | 0.1348 | 0.2011 | 0.9413 | 0.7046 | 0.8059 | -- | 2 (chance floor) | -- |
| per-clip (fixed terms) | 2.0 | 0.1842 | 0.1367 | 0.2013 | 0.9378 | 0.7208 | 0.8151 | 17 | 6 | **+13** |
| per-clip (fixed terms) | 3.0 | 0.1845 | 0.1367 | 0.2018 | 0.9378 | 0.7208 | 0.8151 | 17 | 10 | **+9** |

Compare to `2026-09-02-per-clip-also-regresses.md`'s numbers on the same
4 clips, buggy extractor: **net -4** at cb=2.0, **net -9** at cb=3.0,
**terms_recovered=0 at every weight tested**. Fixing the extractor flips
the sign at both weights, and moves `terms_recovered` from a flat zero to
17.

The recovered terms are almost entirely one family: "Monro Forward" (a
program name on Monro's own call), correct 0/14 times unbiased and 13/14
times biased at cb=2.0, plus three related multi-word variants
("Monro Forward Initiatives", "Monro University", "Monro Inc Earnings
Conference Call") each going from 0 correct to 1-4. This is the textbook
case biasing is supposed to fix: a genuinely rare, company-specific term
the baseline model is confidently wrong about every single time it's
said, recovered by giving the model a hint it didn't have.

Per-clip WER: 3 of 4 biased clips get marginally worse (Monro 0.1581 ->
0.1651, Pilgrim's 0.1535 -> 0.1549, Eversource 0.2105 -> 0.2107); GE
improves slightly (0.1800 -> 0.1795). Injected terms at cb=2.0: `Dollars,
Steve Winoker, tender, U.S., lower earnings, Electric Distribution` --
still real false accepts, not eliminated by the fix, just outweighed by
the Monro Forward recovery in the net accounting.

**Whole-transcript WER still does not beat the unbiased baseline at
either weight** (0.1842/0.1845 vs. 0.1835) -- the same headline finding
as before survives. What changed is the term-level story underneath it:
"pure cost, zero recovery" is no longer true. There is a real recovery
happening, it is being netted against real remaining false accepts, and
the net of those two is now positive rather than negative.

## Why this run's raw WER numbers aren't directly comparable to the old doc's

This run's baseline WER is 0.1835; `per-clip-also-regresses.md`'s was
0.1773. Same model, same 4 biased clips, same unbiased decode -- these
should be identical if the clip pool were the same. It wasn't: this run's
`clip_dir` now holds all **11** Earnings-21 clips (the 4 with real term
lists plus 7 that never got past SEC lookup/8-K search/pyate's
`OverflowError`, added to the shared eval directory sometime between
the two per-clip runs, most likely by the later oracle/distractor work
which needs all 11). `jiwer.wer` pools edit distance across the whole
list, so the 7 empty-term-list clips -- which get byte-identical
baseline and biased output, since an empty context graph spots nothing
-- dilute the aggregate WER/B-WER numbers with unrelated, unaffected
text and also shift the baseline itself by mixing in 7 additional clips'
worth of difficulty. **`terms_recovered`, `injections_added`, and
`net_term_gain` are unaffected by this** -- those are computed per-term
from `per_term_stats`, and the 7 empty-list clips contribute exactly
zero delta to any of them in either run, since nothing about their
output changes between baseline and biased. The net-term comparison
above is the reliable one; the absolute WER figures either side of it
are two different eval-set sizes and shouldn't be diffed directly.

## Consequence for Task 4 (the headroom signal)

This doesn't make the headroom-signal work unnecessary. It changes what
question it's answering: not "can anything be salvaged from a pipeline
that recovers zero terms," but "can real remaining false accepts (6-10
per weight, non-generic ones like `Steve Winoker` and `Electric
Distribution` that survived the extraction filter because they're
genuinely rare, just not said in that clip) be told apart from genuine
recoveries like Monro Forward, on the same clip, before deciding whether
to boost." That is arguably the more honest version of the question --
validating a headroom proxy against a run still dominated by extraction-bug
noise would have risked confounding "does this catch bad terms" with
"does this catch bad extraction." `docs/superpowers/specs/2026-09-03-headroom-signal-validation-design.md`'s
retrospective validation should use this run
(`data/eval_results/sweep_earnings21_perclip_v2/`) as its ground truth,
not the superseded one.

## Limits

Still 4 biased clips -- n=4 was already flagged as underpowered in the
original doc and remains so; a single clip's outcome can move `net` by
several units, and this rerun did not add clips, only fixed what feeds
them. The 7-clip SEC-pipeline gap (CIK misses, missing 8-K, pyate
`OverflowError`) is unfixed and unrelated to this bug; those clips still
contribute no bias signal either way. Whether the extraction fix
generalizes beyond these 4 companies' filings is unmeasured.

Full data: `data/eval_results/sweep_earnings21_perclip_v2/sweep-summary.json`
and the per-weight `sweep-per-clip-{2.0,3.0}.json` files (per-clip
predictions and injected-term lists).
