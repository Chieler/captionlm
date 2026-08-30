# Domain-Biased Captioning — Design

Date: 2026-08-29
Status: approved (architecture + phases), pending implementation plan

## Problem

Generic ASR mistranscribes domain-specific terms (product names, jargon,
acronyms) that are rare in the model's training data but known in advance —
already extracted into a term list by a separate process (not in this repo's
scope). This system biases transcription toward that known term list without
retraining or slowing down decoding, following the CTC-based Word Spotter
(CTC-WS) method from Andrusenko et al., *Fast Context-Biasing for CTC and
Transducer ASR with CTC-based Word Spotter* (NVIDIA, Interspeech 2024,
arXiv:2406.07096).

## Method summary

CTC-WS runs a token-passing search over CTC log-probabilities against a
compact trie ("context graph") of the known terms' subword tokens. Matches
above a score threshold become "spotted word" hypotheses with frame-level
start/end. These are merged into the model's normal greedy/transducer output
by replacing whatever word(s) occupy the same frame span, whenever the
spotted word's score beats the frame-aligned CTC score for that span. Because
the search runs on log-probs the encoder already produced, cost is
independent of encoder size and this is what gives the "significant
acceleration" over decode-time biasing methods — no beam search, no model
modification, no retraining.

CTC-WS requires a CTC head. A pure transducer (TDT-only) checkpoint has none.
NVIDIA's paper solves this with a hybrid Transducer-CTC model: shared
encoder, two decoder heads, spotting runs against the CTC head's log-probs,
merge target is the transducer's (TDT) hypothesis — TDT is the more accurate
output; CTC/spotting is only ever used to correct it, never to replace it
wholesale.

## Scope (this phase)

Offline batch captioning of local video/audio files. Live-mic captioning is
explicitly deferred but the core (`generate()` override on a stateless
per-chunk model) must not preclude it — see "Live-mic seam" below.

## Non-goals

- Live/streaming capture (future work, seam only)
- Term extraction from documents (already done upstream; input is a
  plain-text file, one term per line)
- Training or fine-tuning any model
- GPU/CUDA — target is Apple Silicon unified memory, MLX runtime

## Architecture

```
term list (given, .txt) ──sentencepiece(tokenizer.model)──► ContextGraphCTC   [built once]
                                                                  │
video/audio file ──ffmpeg (via parakeet-mlx)──► samples 16k       │
        │                                                        │
        └─► model.transcribe(path, chunk_duration=...)  ◄─────────┘   [parakeet-mlx, unmodified]
                    │ chunks long files, overlap-merges results per chunk
                    ▼
            BiasedParakeetTDTCTC.generate(mel)              [ours]
                    │
              features, lengths = self.encoder(mel)          ← the only expensive step
                    ├─► self.ctc_decoder(features) ──► logprobs [T, V+1]
                    │        └─► run_word_spotter (vendored) ──► WSHyp[]
                    ├─► self.decode(features, lengths) ──► AlignedToken[]  (TDT greedy)
                    └─► merge_spans(tokens, hyps, time_ratio)  [ours]
                    ▼
            tokens_to_sentences ──► AlignedResult             [parakeet-mlx, unmodified]
                    ▼
              .srt / .vtt output                              [parakeet-mlx, unmodified]
```

### Why subclass instead of a standalone pipeline

`ParakeetTDT.generate()` is five lines: encoder → decode → sentence
grouping. Subclassing `ParakeetTDTCTC` and overriding only `generate()`
inserts CTC-decode + spot + merge between two existing steps. Everything
above (`transcribe()`'s chunking, cross-chunk overlap merge via
`merge_longest_contiguous`) and below (sentence grouping, `.srt`/`.vtt`
writers) is reused unmodified. A standalone pipeline calling the encoder
directly would require reimplementing all of that — rejected as
reinventing already-correct code.

### Modules

| File | Purpose | New/vendored |
|---|---|---|
| `captionlm/terms.py` | load term list, sentencepiece-encode, build `ContextGraphCTC` | new |
| `captionlm/biased_model.py` | `BiasedParakeetTDTCTC(ParakeetTDTCTC)`, `generate()` override | new |
| `captionlm/merge.py` | span-replacing merge of `WSHyp` into `AlignedToken` sequence | new |
| `captionlm/config.py` | model id, spotter hyperparameters (`cb_weight`, `beam_threshold`, etc.) | new |
| `captionlm/cli.py` | CLI: file/dir argument → captioned output | new |
| `captionlm/eval.py` | F-score / WER, biasing on vs off, `cb_weight` sweep flag | new |
| `captionlm/vendor/context_graph_ctc.py` | trie structure for context graph | vendored from NeMo, Apache-2.0, unmodified |
| `captionlm/vendor/ctc_word_spotter.py` | token-passing search (`run_word_spotter`) | vendored from NeMo, Apache-2.0, 2-line diff (tokenizer→vocabulary list) |
| `captionlm/vendor/context_biasing_utils.py` | `compute_fscore` (eval metric) | vendored from NeMo, Apache-2.0, unmodified — also contains `merge_alignment_with_ws_hyps`, NOT used (we use our own `merge.py` since parakeet-mlx's types differ from NeMo's `rnnt_utils.Hypothesis`) |

Each vendored file keeps a header comment with the NeMo source path and
commit/date pulled from, so it can be re-diffed on upstream changes.

### Data contracts

- **Term list input**: plain text file, one term per line (UTF-8, blank
  lines and `#`-prefixed comments ignored).
- **Multi-word terms**: supported. `ContextGraphCTC` builds a trie over BPE
  token sequences regardless of how many words a term spans; `merge.py`
  must therefore replace a *span* of one or more transducer words per spot,
  keyed on frame-range overlap ≥ the configured threshold (default 30%,
  matching NeMo), not a single word.
- **CLI input**: explicit file or directory argument (audio/video,
  whatever `ffmpeg`/parakeet-mlx already accepts). Fully independent of
  `file_import.py` and `static_assets/`, which handle document text
  extraction for a different purpose (building the term list upstream, out
  of scope here).
- **Eval input**: a directory of `<clip>.wav` + `<clip>.txt` (reference
  transcript) pairs. Empty at first — this phase ships the harness, not
  the data; you populate it after the pipeline works.

### Correctness-critical details

- **Blank index.** parakeet-mlx's `ConvASRDecoder` appends blank as the
  *last* class (`num_classes = len(vocabulary) + 1`). `run_word_spotter`
  and `merge.py` must be called with `blank_idx=len(vocabulary)`, not the
  vendored default of `0`. Getting this wrong doesn't crash — the spotter
  silently spots nothing. This is asserted in the Phase 2 self-check.
- **Vendored tokenizer shim.** NeMo's spotter calls
  `asr_model.tokenizer.ids_to_tokens(...)`; parakeet-mlx exposes a plain
  `vocabulary: list[str]`. The 2-line diff replaces those calls with
  `vocabulary[id]` lookups. No other vendored logic changes.
- **Frame/time conversion.** CTC logprobs are indexed in encoder frames;
  `AlignedToken.start` is in seconds. Convert via `model.time_ratio`
  (`round(seconds / time_ratio)`), matching NeMo's own frame math.

### Live-mic seam (not built this phase)

`generate()` is called once per chunk and is otherwise stateless aside from
reading `self.ctc_decoder` / `self.decode`. `StreamingParakeet` (already in
parakeet-mlx) dispatches per-chunk decode calls by model type. Extending to
live audio later means teaching that streaming path to also call the CTC
head and `merge_spans` per chunk — no change to `terms.py`, `merge.py`, or
the vendored spotter is anticipated.

## Dependencies

- `parakeet-mlx` — MLX runtime, chunking, alignment types, decoders
- `sentencepiece` — encode term list against the shipped `tokenizer.model`
- `jiwer` — WER computation for `eval.py` (MIT license, pure Python, no
  torch/CUDA — matches the low-footprint goal)
- `numpy` — vendored spotter's only dependency

Explicitly not adding: `nemo_toolkit` (torch + CUDA-shaped, ~2GB, wrong fit
for Apple Silicon background use — this is why the spotter/graph/fscore
code is vendored instead of imported).

## Model

`mlx-community/parakeet-tdt_ctc-110m`, set as a `config.py` constant. Only
hybrid TDT-CTC checkpoints work with this method (pure-TDT checkpoints like
`parakeet-tdt-0.6b-v2` have no CTC head). 110m chosen for footprint
(~0.25GB bf16); `1.1b` is a one-line config change if the eval harness
shows the accuracy gap matters enough to justify ~2.4GB resident.

## Testing / verification

- **Phase self-checks** (per writing-plans convention): graph roundtrip
  assert in `terms.py`, blank-index assert in `biased_model.py`, a merge
  unit test with a synthetic single-word and a synthetic multi-word overlap
  case in `merge.py`.
- **Phase 4 eval harness** is the real verification: F-score (vendored
  `compute_fscore`) and WER (`jiwer`) computed with biasing on vs off over
  the labeled clip set, and a `--cb-weight` sweep flag so tuning is a
  rerun, not a code edit. This is deferred until clips exist (5–20,
  user-collected after the pipeline runs) but the harness itself ships in
  this phase.

## Phases

1. **Env + vendor** — add `parakeet-mlx`, `sentencepiece`, `jiwer` to
   venv; pull the three NeMo files into `vendor/`, apply the tokenizer
   shim diff, add source-commit comments.
2. **Term graph** — `terms.py` + `config.py`. Self-check: known term
   roundtrips through the graph.
3. **Biased model + merge** — `biased_model.py`, `merge.py`. Self-check:
   blank-index assert, merge unit test (single-word + multi-word span).
4. **CLI** — `cli.py`, file/dir argument → `.srt`/`.vtt` output using the
   biased model.
5. **Eval harness** — `eval.py`: F-score/WER on vs off, `cb_weight` sweep
   flag. Ships with empty clip directory; data comes later.

## Open questions

None outstanding — all resolved during brainstorming (term format, eval
data timing, CLI input path, model tier, git init).
