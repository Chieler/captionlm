# Handoff

## Current product behavior

captionlm is a local, document-aware transcription tool. The product paths
(`scripts/serve.py` and `scripts/caption_dropoff.py`) now share the same
policy as `captionlm.cli`:

- no matching document: normal transcription, no contextual bias;
- matching document, at most 300 seconds: context bias at `cb_weight=3.0`;
- matching document, over 300 seconds: conservative context bias at
  `cb_weight=1.0`.

The selector is reset for every recording because models are cached across a
batch. A long file must not make the following short file conservative.

## Evidence

Read [RESULTS.md](RESULTS.md) and [ARCHITECTURE.md](ARCHITECTURE.md) before
changing model selection or bias policy. The reliable conclusion is narrow:
document bias strongly helps genuinely rare, spoken terms, but can trade some
ordinary-word accuracy for those gains on long spontaneous speech.

## Next work

1. Finish per-clip Earnings-21 coverage. The term-extraction counter overflow
   and several historical CIK aliases are fixed; foreign 6-K selection still
   needs a content-aware, tested selector.
2. Compare unbiased versus `cb_weight=1.0` on the expanded long-form set.
   The cap is a safety stopgap, not a proven optimum.
3. Evaluate a plain, no-document mode on lecture-like speech. Do not build a
   learned router until it wins prospectively on WER, B-WER, U-WER, recoveries
   and injections.

## Before evaluation

```bash
venv/bin/python -m pytest -q
venv/bin/python -c 'import mlx.core as mx; print(mx.arange(3))'
```

Four failures in `tests/test_biased_model.py` together point to a dead Metal
compiler service, not a source regression; reboot before debugging code.
