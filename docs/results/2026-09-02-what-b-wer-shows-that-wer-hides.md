# What B-WER shows that WER hides

Overall WER cannot answer the question this project exists to ask. Listed
terms are a minority of the words -- 40% of the read-aloud reference, 27%
of Earnings-21 -- so recovering all of them moves WER by less than the gap
between two configurations, and a regression on ordinary words caused by
the biasing sits invisible underneath the gain.

`captionlm/eval.py:bias_wer` splits the same errors in two. Errors are
attributed by **reference** word: a substitution or deletion of a listed
word is a B-WER error whatever came out instead. Insertions have no
reference word, so they are attributed by the **inserted** word, which is
what stops B-WER from being gamed by emitting listed terms everywhere.
Each rate divides by its own share of the reference words, so B-WER and
U-WER do not average back to WER -- read the pair, never the sum.

Multi-word entries enter the vocabulary as their constituent words, so
`DERIK DE BRUIN` makes `de` a listed word everywhere it appears. That
over-counts, but the alignment is per word and every published B-WER does
the same; a span-level metric would not be comparable to anyone else's
number.

## Read-aloud: the spotter works, and 3.0 is the real optimum

`parakeet-tdt_ctc-1.1b`, each clip biased with its own extracted term
list -- the configuration `cli.py` ships. 1517 reference words, 603 of
them listed.

| cb_weight | WER | B-WER | U-WER | net terms |
|---|---|---|---|---|
| unbiased | 0.1009 | 0.1343 | 0.0788 | -- |
| 0.5 | 0.1015 | 0.1360 | 0.0788 | +1 |
| 1.0 | 0.0943 | 0.1177 | 0.0788 | +24 |
| 1.5 | 0.0844 | 0.0929 | 0.0788 | +54 |
| 2.0 | 0.0831 | 0.0862 | 0.0810 | +60 |
| **3.0** | **0.0784** | **0.0763** | 0.0799 | +67 |
| 4.0 | 0.0798 | 0.0779 | 0.0810 | +69 |
| 5.0 | 0.0824 | 0.0829 | 0.0821 | +71 |
| 6.0 | 0.0949 | 0.1095 | 0.0853 | +58 |
| 8.0 | 0.1358 | 0.1393 | 0.1335 | +37 |

B-WER falls 43% (81 errors to 46) while U-WER moves by one error in 914.
The curve is a genuine U: it bottoms at 3.0 and degrades in both
directions, so NeMo's stock default is right for this set rather than
merely untested. The earlier sweep stopped at 3.0 and could not have
shown that -- it reported monotone improvement to the top of its range.

Two things the split catches that the older metrics do not:

**Term yield and word accuracy disagree past the optimum.** `net` keeps
climbing to +71 at cb=5.0 while B-WER is getting worse (0.0763 to
0.0829). The spotter lands more listed occurrences while mangling the
words around them. B-WER sees it because it counts inserted listed words
as errors; `net` does not, because a recovered occurrence is a recovered
occurrence.

**Over-biasing stops being a term problem and becomes a transcript
problem.** At cb=8.0 U-WER goes 0.0799 to 0.1335. The boost is rewriting
ordinary words wholesale.

What read-aloud *cannot* show is injection. Every row reads `inject=n/a`:
the document is the transcript, so there are no listed-but-unspoken terms
and no denominator for hallucinating one. This set can detect
over-biasing through U-WER, but it can never detect a term list that is
wrong.

## Earnings-21: the split localises the damage

Same model, 9 clips, ~27000s. Every clip biased with Earnings-21's own
published distractor list (1782 entries: 1013 oracle words plus ~769
Fortune 500 names and CEOs known *not* to appear). This is a benchmark
stressor, **not** the shipped configuration -- one list applied to every
clip, deliberately poisoned. No product number may be quoted from it.

| cb_weight 3.0 | WER | B-WER | U-WER |
|---|---|---|---|
| unbiased | 0.1840 | 0.1310 (2459/18772) | 0.2034 (10477/51521) |
| biased | 0.1865 | 0.1302 (2445/18772) | 0.2070 (10665/51521) |

Biasing fixes 14 listed-word errors and creates 188 ordinary-word errors.
All nine clips regress. The injection rate goes 12/15691 to 129/15691 --
ten times the chance floor -- and the yield is net **-65** occurrences.

Overall WER moved 0.184 to 0.187, which alone reads as noise. The split
shows it is not noise but a specific one-directional trade going the
wrong way: B-WER is flat *while* injections rise tenfold, which is the
signature of a list problem rather than a weight problem. The boost is
landing on entries that were never going to be said.

The worst false accepts name the mechanism: `target(24) operations(20)
gap(19) pay(15) jan(15)`. `target` and `gap` are distractor-only -- pure
list poisoning. `operations`, `pay` and `jan` are in the oracle list,
meaning they are genuinely spoken *and* genuinely ordinary words. Nothing
in the pipeline treats a term differently for being common English.

## Open: oracle against distractor

The measurement that separates "biasing is fragile" from "that list was
poisoned" is the same weights run against the oracle list alone, scored
against the distractor vocabulary so the Fortune 500 false accepts stay
visible. It has not been made. The sweep completed its shared baseline
and one cell -- `oracle cb=0.5`, inert at WER 0.1841, B-WER 0.1310,
net -4 -- before the process was lost. cb=0.5 is inert on both eval sets,
so the information is all at 1.0 and above.

Also unmeasured: the `per-clip` condition on Earnings-21, which is the
shipped path on spontaneous speech. Until that exists, nothing here says
whether captionlm helps or hurts on real spontaneous audio -- only that a
hostile list hurts.

## Limits

Read-aloud is two clips and 1517 words. A 43% B-WER improvement is 35
errors. The direction is unambiguous; the magnitude has a wide interval
and should not be quoted as a specification.
