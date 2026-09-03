# A router needs a better headroom proxy than transcript presence

`parakeet-tdt_ctc-1.1b`, full-clip batch decode (`chunk_duration=120s`).
Two `cb_weight`-selection strategies benchmarked against each other across
three regimes already measured this session: Earnings-21 per-clip (4
clips, real SEC term lists, long audio), LibriSpeech `test-clean` N=1000
(100 clips), LibriSpeech `test-other` N=1000 (100 clips). The question:
given that the optimal weight clearly differs by regime (`2026-09-02-*`
docs), can a router pick it automatically, and is a second (adaptive)
pass worth its cost over a first (static) one.

## The two strategies

**Approach A (static, single-pass)**: `cb_weight` chosen from `list_size`
and `audio_duration` alone, before any transcription happens.

```
duration > 300s        -> 1.0   (long-form; no static signal for headroom,
                                  and every weight >=2.0 tested regressed
                                  on Earnings-21 per-clip)
duration <= 300s:
  n_terms <= 200        -> 3.0  (matches N=100's forgiving range)
  n_terms > 200          -> 2.0  (matches the N>=500 optimum found for
                                  both LibriSpeech splits)
```

**Approach B (adaptive, two-pass)**: run an unbiased pass first, then
override toward `0.5` when a ground-truth-free proxy suggests baseline
already knows the list -- `self_recall >= 0.85`, where `self_recall` is
the fraction of the clip's own term list found (whole-word match) in the
*unbiased* transcript. Otherwise falls back to Approach A's weight. The
self-recall check is legitimately available at inference time (no
reference text used); the intent was to catch exactly the Earnings-21
per-clip pathology (`2026-09-02-per-clip-also-regresses.md`: baseline
already gets 97-100% of genuinely-spoken listed terms right).

## Results

| regime | approach | weight(s) chosen | WER | B-WER | U-WER | net | biased-pass time | total runtime |
|---|---|---|---|---|---|---|---|---|
| Earnings-21 per-clip | A | 1.0 | 0.1776 | 0.1396 | 0.1916 | -2 | 965s | 965s |
| Earnings-21 per-clip | B | 1.0 | 0.1776 | 0.1396 | 0.1916 | -2 | 642s | 1230s (+588s baseline) |
| LibriSpeech clean N1000 | A | 2.0 | 0.0154 | 0.0171 | 0.0113 | +6 | 54s | 54s |
| LibriSpeech clean N1000 | B | 2.0 | 0.0154 | 0.0171 | 0.0113 | +6 | 53s | 105s (+52s baseline) |
| LibriSpeech other N1000 | A | 2.0 | 0.0395 | 0.0429 | 0.0335 | +9 | 56s | 56s |
| LibriSpeech other N1000 | B | 2.0 | 0.0395 | 0.0429 | 0.0335 | +9 | 54s | 272s (+218s baseline) |

**Approach B never chose a different weight than Approach A, on any clip,
in any regime.** Identical WER/B-WER/U-WER/net in every row above --
paired exactly, not approximately. B's only measured effect is cost:
27% more wall-clock on LibriSpeech clean, 27% more on Earnings-21
per-clip's small-N case, ~390% more on LibriSpeech other (baseline pass
was disproportionately expensive there). As tested, B is strictly
dominated by A: same accuracy, more compute, no exceptions in this run.

## Why the self-recall gate never fired

`self_recall` is computed over the **whole term list**, but on Earnings-21
per-clip only ~37% of an extracted list is ever actually spoken in that
clip (`unspoken_terms=126/200` from `2026-09-02-per-clip-also-regresses.md`).
A term that was never said cannot appear in *any* transcript, biased or
not -- so the denominator is dominated by terms that are structurally
unrecoverable, capping the ratio near the spoken fraction (~19/50 = 0.38
for one specific clip, checked directly) regardless of how well baseline
handles the words that were actually said. **Without ground truth,
"term absent from the unbiased transcript" is ambiguous between "wasn't
said" and "was said but missed,"** and real per-clip term lists (extracted
from a whole document, most of which doesn't come up in any one call) are
dominated by the first case. `self_recall` as defined here measures list
coverage, not baseline competence -- a different quantity, and the two
diverge exactly whenever most of the list goes unspoken, which is the
normal case for this product's real per-clip usage, not an edge case.
This is a design flaw in the proxy, not an implementation bug: the code
computes exactly what it was asked to, and what it was asked to compute
turned out not to isolate the thing it needed to.

## What did work: static duration capping, for free

Approach A's single rule (`duration > 300s -> cb_weight = 1.0`) cuts
Earnings-21 per-clip's harm from **net -9** (the shipped `cb_weight=3.0`
default, `2026-09-02-per-clip-also-regresses.md`) to **net -2** -- better
than the `ctc_ali_token_weight=1.5` ablation measured earlier the same
day (net -4). That ablation was reported in-session but never written up
as its own `docs/results/` entry -- the raw numbers live in this
session's transcript and in `data/eval_results/sweep_earnings21/`'s
manually-run ablation JSONs, not in a committed doc; flagging that gap
here rather than inventing a citation for it. The duration-capping fix
found today is cheaper regardless: no change to spotter internals, just a
different default for one config field, chosen from information (audio
duration) that's free and already known before transcription starts. It doesn't
fully close the gap -- WER 0.1776 vs. baseline's 0.1773, a hair worse
still -- but it recovers most of the available improvement at zero
additional engineering or runtime cost over what ships today.

## What a working adaptive proxy needs

Not whole-transcript presence -- per-span confidence at the specific
locations a listed term might plausibly occur. The word spotter's own
raw pre-filter candidates (instrumented in this session's diagnostics
run, `captionlm/vendor/ctc_word_spotter.py`'s `find_best_hyps` input)
already carry timestamps; a real headroom signal would check baseline's
own decode confidence at those specific spans, which can distinguish
"audio here is genuinely uncertain" from "term wasn't spoken here at
all, irrelevant to headroom." That is a materially bigger build than
what this benchmark ran and was not attempted here.

## Limits

Weight-selection rules were designed from this session's own prior
sweep results (a handful of points per regime), not fit to held-out
data -- Approach A's thresholds (300s, 200 terms, the specific weight
values) are defensible from what's measured but not independently
validated. Only one self-recall threshold (0.85) was tested for Approach
B; the finding here is that the proxy itself is unsound at any threshold
(it structurally can't separate "unspoken" from "spoken-but-missed"),
not that 0.85 was the wrong cutoff. Runtime numbers are wall-clock on an
otherwise-idle machine, comparable within this session but not a
controlled benchmark against other hardware or load conditions.

Raw output: `router_earnings21.json`, `router_librispeech_clean.json`,
`router_librispeech_other.json` (scratchpad, not committed to the repo).
