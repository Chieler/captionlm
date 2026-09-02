# Handoff: continuing the B-WER work on another machine

Written 2026-09-02. Read `docs/results/2026-09-02-what-b-wer-shows-that-wer-hides.md`
first -- it is the measurement record and this file only says what to do next.

## Where things stand

`captionlm/eval.py:bias_wer` splits WER into B-WER (reference words on the
term list) and U-WER (everything else). It is wired into both arms of
`score()` and printed by `captionlm.eval` and `scripts/sweep_cb_weight.py`.

Settled:

- **The spotter works.** Read-aloud, `parakeet-tdt_ctc-1.1b`, per-clip term
  lists: B-WER 0.1343 -> 0.0763 at cb_weight 3.0, U-WER 0.0788 -> 0.0799.
- **cb_weight 3.0 is a real optimum**, bracketed on both sides -- the curve
  degrades below 2.0 and above 4.0, and at 8.0 U-WER nearly doubles.
- **Earnings-21 with the published distractor list is net-negative** at 3.0:
  B-WER flat (0.1310 -> 0.1302), U-WER worse (0.2034 -> 0.2070), injections
  10x the chance floor, net -65 term occurrences, all 9 clips regress.

Open, in priority order. Neither has been measured.

## Task 1: oracle against distractor (~1h, decides the next month of work)

Earnings-21's distractor list is the oracle list plus ~769 Fortune 500 names
and CEOs that are never spoken. At cb_weight 3.0 the worst false accepts were
`target(24) operations(20) gap(19) pay(15) jan(15)`. `target` and `gap` are
distractor-only; `operations`, `pay` and `jan` are in the oracle list, so
they are genuinely spoken *and* genuinely ordinary English words.

Running both lists at the same weights, scored against the same vocabulary,
separates two very different diagnoses:

- **Oracle arm also net-negative** -> biasing itself is fragile on
  spontaneous speech. The method needs work, not the list.
- **Oracle arm net-positive, distractor arm negative** -> the damage is
  unspoken list entries. Fix term extraction and add a common-word guard;
  the method is fine.

```bash
venv/bin/python scripts/sweep_cb_weight.py data/eval_data/bias_clips \
    --condition oracle=data/speech-datasets/earnings21/bias_lists/oracle_list.txt \
    --condition distractor=data/speech-datasets/earnings21/bias_lists/distractor_list.txt \
    --score-terms data/speech-datasets/earnings21/bias_lists/distractor_list.txt \
    --weights 1.0,2.0,3.0 \
    --model mlx-community/parakeet-tdt_ctc-1.1b \
    --out-dir data/eval_results/sweep_earnings21
```

Both conditions score against the **distractor** vocabulary on purpose.
Scoring the distractor run against the oracle list would silently drop every
distractor false accept, which is the harm the run exists to measure.

## Task 2: the shipped path on spontaneous speech (~35 min)

Every Earnings-21 number so far applies one benchmark list to all nine clips.
That is not what `cli.py` ships, and `scripts/sweep_cb_weight.py`'s own
docstring says no product number may be quoted from it. The shipped path is
one document biasing its own audio -- the `per-clip` condition.

`data/eval_data/bias_clips/` has no `<clip>.terms.txt`, so build them first
(`captionlm/build_eval_set.py` does SEC filing lookup; `scripts/extract_terms.py`
turns any document into a list). Then:

```bash
venv/bin/python scripts/sweep_cb_weight.py data/eval_data/bias_clips \
    --condition per-clip=per-clip \
    --score-terms data/speech-datasets/earnings21/bias_lists/distractor_list.txt \
    --weights 2.0,3.0 \
    --model mlx-community/parakeet-tdt_ctc-1.1b \
    --out-dir data/eval_results/sweep_earnings21_perclip
```

Until this exists, nothing establishes whether captionlm helps or hurts on
real spontaneous audio -- only that a deliberately poisoned list hurts.

## Task 3: common-word guard (only if Task 1 blames the list)

Nothing in the pipeline treats a term differently for being frequent English.
`operations`, `pay` and `jan` inject even from the oracle list. A frequency
filter or per-term weight in `captionlm/terms.py` is the obvious lever. Do not
build it before Task 1 says the list is the problem.

## Before you run anything

```bash
venv/bin/python -m pytest -q                          # expect 135 passed
venv/bin/python -c "import mlx.core as mx; print(mx.arange(3))"
```

The MLX smoke test is not paranoia. On the origin machine the Metal compiler's
XPC service died mid-session and every MLX call started failing with
`Compiler encountered XPC_ERROR_CONNECTION_INVALID`. The symptom is exactly
three failures in `tests/test_biased_model.py` while the other 132 tests pass,
because those are the only tests that touch the GPU. It killed a running sweep
and it is fixed by a reboot, not by editing code. If you see those three
failures, stop and reboot -- do not debug them.

## Traps that cost time on the origin machine

- **`cb_weight` 0.5 is inert** on both eval sets -- WER, B-WER and U-WER all
  unchanged to four decimals. Do not spend a 25-minute pass on it.
- **Long runs die with the terminal.** Use
  `nohup ... > run.log 2>&1 &` and poll the log.
- **`pgrep` lies when `sysmond` is down** -- it exits non-zero with
  `Cannot get process list`, which a `while pgrep` watcher reads as "finished".
  Use `kill -0 <pid>` or `ps -p <pid>` instead.
- **The sweep's unbiased baseline is one shared ~21-minute pass** reused by
  every cell. Killing the sweep forfeits it; there is no resume.
- **Budget ~16 min per pass** for the 9 Earnings-21 clips on an unloaded
  machine, roughly 25 under contention. A 2-condition x 3-weight grid is
  ~2 hours including the baseline.

## Do not

- **Do not quote the Earnings-21 distractor numbers as product numbers.**
  They come from a hostile list applied uniformly to every clip.
- **Do not read read-aloud's monotone improvement as proof a weight is safe.**
  Every row there reads `inject=n/a`: the document *is* the transcript, so
  there are no listed-but-unspoken terms and no denominator for hallucinating
  one. That set can detect over-biasing through U-WER but can never detect a
  wrong term list.
- **Do not restart the live/streaming work** without addressing its own gate.
  It failed GATE 2 at +5.8-6.5pp live-vs-batch, ~3x over threshold, and the
  cause is structural: commit-boundary deletions plus a 30s context window
  against batch's 120s. See the live-streaming plan under `docs/superpowers/plans/`.

## Getting the data

`SETUP.md` has the full sequence. Short version: read-aloud `.wav` and
`.terms.txt` are tracked, so that set runs straight after install.
Earnings-21 is 1.7 GB and is not in git -- clone
`revdotcom/speech-datasets` into `data/speech-datasets/` and run
`scripts/build_bias_clips.py`.
