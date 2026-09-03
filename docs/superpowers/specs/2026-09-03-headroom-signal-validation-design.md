# Headroom signal validation — design

Does the model's own confidence at a specific point in the audio predict
whether biasing a term there will help or hurt? Nothing in this codebase
answers that yet. `docs/HANDOFF.md`'s Task 4 names the question; this is
the first attempt to actually measure it, cheaply, before trusting it.

The reason this can't just be built straight into the router: the
previous attempt at a headroom proxy (`self_recall`, whole-transcript
term presence) also looked reasonable on paper and turned out to be
structurally unsound once measured
(`docs/results/2026-09-03-a-router-needs-a-better-headroom-proxy.md`).
This spec is scoped to not repeat that mistake — it is a measurement, not
a feature. Nothing here is wired into `cli.py`.

**Ground truth for validation is `data/eval_results/sweep_earnings21_perclip_v2/`,
not the older `sweep_earnings21_perclip/`.** The original per-clip sweep
was confounded by a term-extraction bug that made every recovered term
look like an injection-only loss (`docs/results/2026-09-02-per-clip-also-regresses.md`);
fixing it flips `terms_recovered` from 0 to 17
(`docs/results/2026-09-03-fixing-term-extraction-flips-per-clips-net-positive.md`).
Validating a headroom proxy against the superseded run would confound
"does this proxy catch bad terms" with "does this proxy catch bad
extraction" — the same failure mode this spec's intro already flags for
`self_recall`. The `_v2` run has real recoveries (the "Monro Forward"
family) alongside real remaining false accepts (`Steve Winoker`,
`Electric Distribution`), which is what makes it worth validating
against: a signal that can't tell those two apart on this data isn't
ready to trust on anything harder.

## The constraint that shapes everything

The signal this needs — the model's own decode confidence at the specific
place a listed term might occur — requires raw CTC logprobs, and nothing
in the pipeline keeps them. `BiasedParakeetTDTCTC.generate()`
(`captionlm/biased_model.py:28`) computes `ctc_logits` internally per
chunk whenever a context graph is set, then discards them the moment the
word spotter has run over them. `model.transcribe()` returns only the
final decoded `AlignedResult`. There is no path today to get the
logprobs back out without either reimplementing the encoder call by hand
(risking silent divergence from what `generate()` actually does — chunk
boundaries, dtype, batch handling) or adding a way to capture what
`generate()` already computes. The second is smaller and safer, so that's
the one piece of production code this touches.

## Approach

**Argmax-alignment gap.** For each term, run the CTC word spotter with
`cb_weight=0.0` against the clip's own unbiased logprobs — this finds the
term's best-matching span using pure acoustic evidence, no artificial
boost. Then measure how confident the argmax-decoded path already is
over that same span, using `get_ctc_word_alignment`
(`captionlm/vendor/ctc_word_spotter.py:151`), which is already vendored
and already trusted for an almost-identical comparison: `filter_wb_hyps`
(same file, line 218) uses it *after* a biased decode to throw out
spotted hypotheses that scored worse than what the argmax path already
had there. This reuses that same comparison, moved earlier — before
deciding whether to bias at all, not after.

    headroom(term) = best_unbiased_spotted_score(term) - argmax_score_at_that_span

A large gap means the term's own acoustic evidence beats what the model
currently, confidently believes is there — real headroom, worth boosting.
A small or negative gap means the argmax path is already about as
confident as the term's own match — either it's already right, or it's
confidently wrong about something else, and either way overriding it is
mostly risk. No surviving candidate span at all is a third, distinct
outcome: no acoustic trace of the term anywhere in the clip (it likely
wasn't said, or the audio doesn't support it regardless of weight).

An alternative (scoring how close an unbiased spot came to surviving beam
pruning on its own, with no argmax comparison at all) was considered and
rejected for this first pass: it has no existing precedent in this
codebase to lean on, where the argmax-comparison approach reuses math
`filter_wb_hyps` already relies on in production.

**Validation is retrospective.** Every eval set this project has already
swept (`data/eval_results/sweep_*`) has per-clip baseline/biased
predictions saved to disk, and the term's actual fate (recovered,
injected, or neither) can be recomputed from that saved text for free —
`captionlm/eval.py`'s `per_term_stats` and `unspoken_injections` already
do exactly this and need no changes. The only work that's new is one
extra transcription pass per already-swept clip, with logit capture
turned on, to get the raw material for the headroom score. That pass
costs the same as re-running one more sweep cell — nothing cheaper was
found, but nothing more expensive either, and it reuses the tested
`transcribe()` path exactly rather than risking a hand-rolled
encoder-only forward pass that quietly computes something different from
what production does.

## Components

### `captionlm/biased_model.py` (modify)

One capture hook, off by default, zero behavior change when unused.

```python
class BiasedParakeetTDTCTC(ParakeetTDTCTC):
    def __init__(self, args: ParakeetTDTCTCArgs):
        super().__init__(args)
        self.context_graph: ContextGraphCTC | None = None
        self.spotter_config = SpotterConfig()
        self.capture_logits: list[np.ndarray] | None = None  # new
```

In `generate()`, inside the existing per-batch-item loop (the `if
self.context_graph is None: ... else:` branch), right after the existing
line `logprobs = np.array(ctc_logits[batch_idx].astype(mx.float32))`:

```python
if self.capture_logits is not None:
    self.capture_logits.append(logprobs.copy())
```

The hook adds a copy of a value the loop already computes on every
iteration — nothing new is calculated. One list entry per internal
chunk, in order; a long Earnings-21 clip produces several, since
`chunk_duration=120.0` splits it into multiple `generate()` batch items.

### `captionlm/headroom.py` (new)

Pure numpy + the existing vendored spotter machinery. No model calls,
no I/O.

```python
def argmax_span_score(
    word_alignment: list[tuple], start_frame: int, end_frame: int
) -> float:
    """Overlap-weighted sum of get_ctc_word_alignment's word scores across
    [start_frame, end_frame]. Same accumulation filter_wb_hyps already
    does internally (captionlm/vendor/ctc_word_spotter.py:239-260),
    factored out as its own function because that one only returns a
    survive/drop decision and this needs the graded value."""


def raw_spot(
    logprobs: np.ndarray,
    context_graph: ContextGraphCTC,
    blank_idx: int,
    cb_weight: float = 0.0,
    beam_threshold: float = 5.0,
    keyword_threshold: float = -5.0,
    blank_threshold: float = 0.8,
    non_blank_threshold: float = 0.001,
) -> list[WSHyp]:
    """The token-passing beam search from vendor/ctc_word_spotter.py's
    run_word_spotter (lines 311-356 there), unchanged, stopping before
    that function's own trailing find_best_hyps + filter_wb_hyps calls.
    Needed because run_word_spotter is not reusable for this: it already
    runs the exact argmax-comparison this module wants to compute
    itself, but as a binary keep/drop, and returns only what survives.
    Verified by direct experiment (see below) that calling
    run_word_spotter(..., cb_weight=0.0) does not give a usable signal --
    it returns empty far too often, because a candidate only survives
    its internal filter by outright beating the argmax path in direct
    score competition with zero boost helping it, which is a much
    stricter and less informative bar than the graded gap this needs.
    """


def argmax_span_score(
    word_alignment: list[tuple], start_frame: int, end_frame: int
) -> float:
    """Overlap-weighted sum of get_ctc_word_alignment's word scores across
    [start_frame, end_frame]. Same accumulation filter_wb_hyps already
    does internally (captionlm/vendor/ctc_word_spotter.py:239-260) --
    first overlapping word counted in full, each subsequent overlapping
    word weighted by its overlap percentage -- factored out as its own
    function because that one only returns a survive/drop decision and
    this needs the graded value it's computed from along the way."""


def unbiased_headroom(
    context_graph: ContextGraphCTC,
    logprobs: np.ndarray,
    asr_model,
    blank_idx: int,
    ctc_ali_token_weight: float = 0.5,
    beam_threshold: float = 5.0,
    keyword_threshold: float = -5.0,
    blank_threshold: float = 0.8,
    non_blank_threshold: float = 0.001,
) -> dict[str, float]:
    """One chunk's headroom score per term in context_graph that was
    spotted at all. A term absent from the returned dict had no
    surviving candidate span in this chunk -- distinct from a
    low-but-present score, and the caller (which sees every chunk in a
    clip) should treat a term missing from every chunk's dict the same
    way: no acoustic trace found anywhere in the clip.
    """
```

Implementation: `raw_spot(logprobs, context_graph, blank_idx,
cb_weight=0.0)` for the unbiased candidates (no `find_best_hyps` overlap
resolution needed here -- a term can legitimately have several disjoint
candidate spans in one chunk, and the max-score step below already
picks the best one), `get_ctc_word_alignment(logprobs, asr_model,
token_weight=ctc_ali_token_weight, blank_idx=blank_idx)` once for the
competing argmax alignment, then for each raw hyp: `hyp.score -
argmax_span_score(word_alignment, hyp.start_frame, hyp.end_frame)`,
keyed by `hyp.word`, keeping the max when a term has more than one
candidate span in the chunk.

**Why `raw_spot` duplicates rather than imports run_word_spotter's loop
body:** that loop isn't factored out as its own function in the vendored
file, only inlined inside `run_word_spotter` ahead of the two calls this
module needs to bypass. `captionlm/merge.py` already sets the precedent
in this codebase for handling that: "a from-scratch reimplementation...
not a port" when the vendored code doesn't expose what's needed in the
right shape, rather than editing the vendored file in place.
`raw_spot` still reuses the vendored `Token`, `WSHyp`, `beam_pruning`,
and `state_pruning` directly -- those are already separately imported
and exercised in `tests/test_vendor.py`, confirming they're meant as
reusable primitives, not implementation-private -- so what's duplicated
is only the loop's control flow, not the underlying data structures or
pruning logic.

This does not modify `captionlm/vendor/ctc_word_spotter.py`.

### `scripts/measure_headroom_signal.py` (new)

CLI, matching `scripts/sweep_cb_weight.py`'s existing conventions
(argparse, `<clip>.wav`/`<clip>.txt`/`<clip>.terms.txt` layout).

```
venv/bin/python scripts/measure_headroom_signal.py <clip_dir> \
    --sweep-dir <dir containing sweep-<condition>-<weight>.json> \
    --condition <name> --weight <cb_weight used in that sweep run> \
    --model mlx-community/parakeet-tdt_ctc-1.1b
```

For each clip: load the model once; set `context_graph` from the clip's
own `.terms.txt` (same as the sweep did); set `capture_logits = []`;
call `model.transcribe(wav, chunk_duration=120.0)` (the decoded text is
discarded — only the captured logits matter); for each captured chunk,
run `headroom.unbiased_headroom(...)`; take each term's max score across
all chunks in the clip.

Cross-reference: read `sweep-<condition>-<weight>.json`'s `per_clip`
entries for this clip (already on disk: `reference`, `baseline_pred`,
`biased_pred`) and recompute that clip's per-term outcome with
`captionlm.eval.per_term_stats` and `unspoken_injections` — unmodified,
existing functions, called on a one-clip list instead of the whole set.

Report, per clip and pooled: for terms bucketed by headroom score
(recovered-by-biasing vs. injected-by-biasing vs. neither), does the
score actually separate them? Print a table; no committed doc yet — this
is `docs/results/`-worthy only once it has an answer, per house rules
("measure before claiming").

## Data flow

    <clip>.wav + <clip>.terms.txt
        -> build_context_graph (existing)
        -> model.transcribe(..., capture_logits=[]) [discard decoded text]
        -> capture_logits: list[np.ndarray], one per chunk
        -> headroom.unbiased_headroom(context_graph, chunk, model, blank_idx) per chunk
        -> per-term max score across chunks

    sweep-<condition>-<weight>.json's per_clip[this clip]
        -> eval.per_term_stats / unspoken_injections (existing, unmodified)
        -> per-term outcome: recovered / injected / neither

    (per-term score, per-term outcome) -> correlation report

## Error handling and edge cases

- **A term with no surviving unbiased candidate in any chunk.** Reported
  as `None`/absent, not zero — a real "no evidence anywhere" case, kept
  distinct from "found a weak candidate, scored it low." Both are
  reasons not to boost, but conflating them would hide whether the
  signal can tell them apart, which is itself worth knowing.
- **A clip in `clip_dir` with no matching sweep JSON entry** (e.g. one of
  the 7 Earnings-21 clips that never got a term list). Skip it and say
  so; there is no ground-truth outcome to correlate against.
- **Multiple chunks disagreeing** (a term's best candidate is strong in
  one chunk, absent in another). Max across chunks, per `unbiased_headroom`'s
  contract above — a single strong match anywhere in the clip is the
  relevant signal, not an average diluted by chunks that never had a
  chance to contain the term.

## Testing

**`headroom.py`, unit, no model or audio needed** — hand-built inputs,
matching the existing style in `tests/test_vendor.py` (tiny
`ContextGraphCTC`, tiny synthetic logprobs arrays):

- `raw_spot` on a single-token term with strong acoustic support returns
  one `WSHyp` with the expected score (no boost applied at `cb_weight=0`).
- `raw_spot` returns nothing for a term whose token sequence never
  survives the blank/non-blank/beam/keyword thresholds anywhere in the
  input.
- `argmax_span_score` on a single fully-overlapping alignment word
  returns exactly that word's score.
- `argmax_span_score` on a span overlapping two alignment words returns
  their overlap-weighted sum, not a plain sum (guards against
  regressing to double-counting if `filter_wb_hyps`'s math is ever
  copied here carelessly).
- `unbiased_headroom` on a synthetic case where the spotted term's score
  clearly exceeds the argmax alignment's competing word returns a
  positive gap for that term.
- `unbiased_headroom` returns no entry for a term whose token sequence
  never survives spotting thresholds anywhere in the input.
- `unbiased_headroom` keeps the max score when the same term is spotted
  in two non-overlapping spans of the same chunk with different scores.

**`biased_model.py` hook, GPU-backed, real model** — `tests/test_biased_model.py`
already tests `generate()` this way (real `load_biased_model`, a silent
wav, no stub), because faking MLX's encoder/CTC-decoder output well
enough to exercise the real method body would be more fragile than the
thing it's testing. This hook lives inside that method body, so it gets
the same treatment — one more GPU test in that file, joining the 3
`docs/HANDOFF.md` already tracks as the Metal-XPC canary:

- `capture_logits` defaults to `None`; a `generate()` call with a context
  graph set produces the same `AlignedResult` shape as before this change
  (regression guard — this must not alter the production decode path).
- Setting `capture_logits = []` before a `generate()` call with a
  context graph set results in a non-empty list of numpy arrays
  afterward.

This makes 4 GPU-touching tests in that file, not 3 — `docs/HANDOFF.md`'s
"exactly three failures means the Metal XPC service died" note needs
updating to four as part of this work, or the next person to hit that bug
won't recognize the count.

**`measure_headroom_signal.py`**: no dedicated pytest coverage, matching
`scripts/sweep_cb_weight.py`'s own precedent (no `test_sweep_cb_weight.py`
exists either) — this is a one-off measurement tool, not product code.

## Order of work

One thing gates everything else, and it's a research question, not an
engineering one: **does the score actually separate recovered terms from
injected ones, on data already in hand?** If it doesn't — if headroom
score turns out uncorrelated with outcome, like `self_recall` did — stop
there and report it; there is no router to build on a signal that
doesn't discriminate. Building `measure_headroom_signal.py` end-to-end on
the four Earnings-21 per-clip clips (small, fast, the domain that
actually matters) is the fastest way to that answer, before spending
effort validating further on LibriSpeech or wiring anything into `cli.py`.

## Out of scope

Any change to `cli.py` or `select_cb_weight()`. Any production router.
Validation beyond Earnings-21 per-clip and (if that looks promising)
LibriSpeech's already-swept N=1000 sets — the oracle/distractor
Earnings-21 sweeps and the N=100/500/2000 LibriSpeech variants are not
re-validated here; if the signal holds up, which of those to check next
is a follow-up decision, not part of this pass. Any attempt to fix
`extract_bias_terms` or the term-extraction pipeline (separate, already
in progress as of this session). Streaming/live audio — this only
concerns the batch `transcribe()` path.
