# Results and limits

All figures below use per-clip term lists: each recording receives only the
terms extracted from its own source document. B-WER measures words on the
term list; U-WER measures all other reference words.

| Set / model / policy | WER | B-WER | U-WER | Interpretation |
|---|---:|---:|---:|---|
| Read-aloud, 1.1B, unbiased | 10.09% | 13.43% | 7.88% | Rare terms are missed. |
| Read-aloud, 1.1B, `cb=3` | 7.84% | 7.63% | 7.99% | 67 term occurrences recovered. |
| LibriSpeech rare words, 1.1B, unbiased | 1.83% | 2.12% | 1.13% | Clean rare-word baseline. |
| LibriSpeech rare words, 1.1B, `cb=2` | 1.54% | 1.71% | 1.13% | Best measured balance. |
| Earnings-21 long-form, 1.1B, unbiased (6 clips) | 18.19% | 14.46% | 18.43% | Familiar financial speech. |
| Earnings-21 long-form, 1.1B, `cb=2` (6 clips) | 18.27% | 12.86% | 18.61% | 73 recoveries, 4 added injections. |

The key trade-off: contextual bias solves real rare-term errors, but in
spontaneous long-form audio it can damage ordinary words enough to raise total
WER. The automatic long-form `cb=1` cap is a lower-risk policy than the old
unconditional `cb=3`; it is not yet validated as the final optimum.

The 110M hybrid model remains the lightweight default. On LibriSpeech rare
words it achieved 2.02% WER at `cb=2`, versus 2.21% unbiased. The 1.1B hybrid
is more accurate but requires about 4 GB; the 110M download is about 438 MB.

## Local benchmark check

The 100-utterance LibriSpeech subsets below use the public IS21 deep-bias
recipe: each utterance has its own 1,000-word rare-word list. This is a useful
contextual-ASR stress test, not a replacement for full-corpus leaderboard
scores. The plain 0.6B-v3 checkpoint is the current local open-model reference;
it cannot accept a CTC-WS context graph.

| Same 100-clip subset / local model | WER | B-WER | U-WER | Decode time |
|---|---:|---:|---:|---:|
| test-clean, 0.6B-v3 plain | 1.88% | 2.19% | 1.13% | 66.3 s / 789.6 s audio |
| test-clean, CaptionLM 1.1B plain | 1.83% | 2.12% | 1.13% | not re-timed |
| test-clean, CaptionLM 1.1B, `cb=2` | 1.54% | 1.71% | 1.13% | not re-timed |
| test-other, 0.6B-v3 plain | 5.72% | 6.84% | 3.74% | 94.9 s / 794.1 s audio |
| test-other, CaptionLM 1.1B plain | 5.58% | 6.55% | 3.87% | 112 s / 794.1 s audio |
| test-other, CaptionLM 1.1B, `cb=2` | 5.25% | 5.82% | 4.25% | 118 s / 794.1 s audio |

On the hard subset, CTC-WS recovered 11 listed-word occurrences and added
three list-derived insertions, for a net +8. It improves overall and B-WER,
but increases U-WER: precisely the trade-off the product guardrails must keep
visible. These short-clip timings include per-file dispatch overhead and should
not be advertised as long-recording real-time speed.

For broader context, NVIDIA reports 1.93%/3.59% WER for 0.6B-v3 and
1.82%/3.67% for the 1.1B hybrid on the *full* LibriSpeech clean/other sets,
under its own leaderboard harness. Those are reference numbers, not directly
comparable to the selected rare-word subset above.

## What not to infer

- Read-aloud has no unspoken terms, so it cannot measure term hallucinations.
- A term list shared across recordings is not the shipped configuration and
  must not be used for product claims.
- The offline headroom score is not a router: independent rare-word checks
  showed it measures acoustic weakness, not safe-to-boost confidence.
- The 0.6B-v3 model has no usable CTC head, so it cannot run the CTC word
  spotter even though it is a competent plain transcriber.
