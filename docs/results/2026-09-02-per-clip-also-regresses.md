# Per-clip -- the shipped configuration -- also regresses, on 4 clips

`parakeet-tdt_ctc-1.1b`, full-clip batch decode (`chunk_duration=120s`),
Earnings-21 eval10. **4 clips**, not 11: `captionlm/build_eval_set.py`'s
real SEC EDGAR pipeline -- CIK resolution, 8-K Item 2.02 filing search,
EX-99.1 exhibit extraction, term extraction -- only succeeded for
`4320211` (Monro), `4341191` (General Electric), `4383161` (Pilgrim's
Pride), `4384964` (Eversource). The other 7 failed for three distinct,
structural reasons, not one fixable bug:

- **No CIK match (4):** Zagg, Nextar Media Group, Earthstone Energy,
  Affimed -- plausibly delisted, acquired, or renamed since these
  2020-2021 calls, so they no longer appear in SEC's current
  `company_tickers.json`.
- **No 8-K found (1):** Constellium -- likely a foreign private issuer
  that files earnings via Form 6-K, which this pipeline's Item-2.02
  8-K search does not look for.
- **A real bug (2):** Ingersoll Rand and Aldeyra both raised
  `OverflowError: The elements provided in the data cannot all be casted
  to the dtype uint16` inside pyate's term extraction on those two
  filings' text. Not investigated here -- filed as a known bug, not
  fixed.

Each of the 4 clips was biased with only its own 50-term list, extracted
from that company's actual earnings-release exhibit (`combined` mode:
pyate terminology + spaCy NER) -- the real product pipeline, not a
shortcut, and no cross-clip vocabulary at all.

## Results

| condition | cb_weight | B-WER | U-WER | WER | terms recovered | injections | net |
|---|---|---|---|---|---|---|---|
| unbiased baseline | -- | 0.1395 | 0.1913 | 0.1773 | -- | 0 | -- |
| per-clip | 2.0 | 0.1396 | 0.1918 | 0.1777 | 0 | 4 | -4 |
| per-clip | 3.0 | 0.1400 | 0.1922 | 0.1781 | 0 | 9 | -9 |

`unspoken_terms=126`: of the 4x50=200 extracted terms, 126 (63%) were
never spoken in their own clip at all -- the extraction is pulling terms
the call doesn't use, independent of what biasing does with them.

No swept point beat the unbiased baseline's WER. **`terms_recovered` is 0
at every weight tested** -- biasing did not fix a single listed-word
error on any clip, at any weight. B-WER does not fall, it rises slightly
(0.1395 to 0.1400) as weight increases. Injections more than double
(4 to 9) over the same range. This is not a small net-negative balanced
against some recovered terms -- there is no recovered-terms column to
balance against. It is pure cost.

## What's different from Task 1, and what isn't

`2026-09-02-oracle-list-regresses-too.md` found the oracle arm
net-negative and attributed part of it to cross-clip contamination: a
shared list boosts one clip's genuine terms (`BOEING`, `CREDIT SUISSE`)
against every other clip's audio, where they were never said. Per-clip
removes that mechanism entirely -- each context graph here has exactly
50 terms, all from that one clip's own company filing, none from any
other clip. The regression persists anyway, which rules out cross-clip
contamination as the sole explanation. Whatever is fragile about
context-graph biasing on spontaneous speech is present even in the
maximally narrow, single-document configuration `cli.py` ships.

What's new here and wasn't visible in Task 1: the injected terms are not
short junk tokens (`LOS`, `TIM`, `GC`) or legitimate-but-misapplied
entities (`BOEING` on the wrong clip). They are generic phrase fragments
-- `levels`, `origin`, `laws`, `lower earnings`, `more information`,
`industrial debt maturities`, `share for the second quarter` -- pulled
out of a press release by `combined`-mode extraction (pyate terminology +
spaCy NER, top 50). These are not distinctive domain vocabulary; they are
ordinary financial-boilerplate phrases that happen to score high enough
to make the cutoff. Boosting the model toward "levels" or "more
information" has no plausible upside and a clear failure mode: the
context graph is not distinguishing terms worth biasing toward from
terms that are just common in this genre of document. That is a
different, additional problem from either the cross-clip contamination
in Task 1 or the common-word false-accepts already documented in the
distractor run -- it is upstream, in what `extract_bias_terms` decides
counts as a "term" at all.

## Limits

Four clips, one weight range (2.0, 3.0 only -- the Task 1 sweep already
showed 1.0 and cb=0.5 are inert or nearly so), no held-out set. This is
the shipped configuration on real audio and real client documents, which
makes it the number that matters most for a product claim -- and also
the smallest, least statistically powerful eval in this project's
results so far. A single clip's outcome could shift `net` by several
units. Treat the direction (no swept point beats baseline, zero terms
ever recovered) as a real finding; do not quote precise magnitudes from
n=4.

The 7-clip gap is not neutral: it removes exactly the companies whose
names are unusual enough that SEC lookup or term extraction choke on
them (a foreign issuer, two overflow-triggering filings, four
delisted/renamed companies) -- plausibly correlated with the same
messiness that makes biasing hard in the first place. A cleaner sample
would need either fixing the CIK/6-K/overflow gaps or picking a
different, more currently-tractable set of Earnings-21 companies.

Full per-weight data: `data/eval_results/sweep_earnings21_perclip/sweep-summary.json`.
