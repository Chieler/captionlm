# Domain-Biased Captioning — Design

Date: 2026-08-29
Status: approved (architecture + phases), pending implementation plan

## Problem

Generic ASR mistranscribes domain-specific terms (product names, jargon,
acronyms) that are rare in the model's training data but known in advance
from an organization's own documents. This system extracts that domain
vocabulary from documents and biases transcription toward it without
retraining or slowing down decoding, following the CTC-based Word Spotter
(CTC-WS) method from Andrusenko et al., *Fast Context-Biasing for CTC and
Transducer ASR with CTC-based Word Spotter* (NVIDIA, Interspeech 2024,
arXiv:2406.07096).

Document text extraction (`.docx`/`.pdf`/OCR) is already solved —
`file_import.py` uses `extractous` (+ Tesseract for OCR, both already
installed) and has a working TIKA-198 workaround for Google-Docs-exported
`.docx` files. That stays as-is. What's missing, and now in scope, is
turning extracted document text into a *domain-specific* term list — not
just any keyword, but vocabulary unusually specific to these documents
relative to general English.

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

Domain term extraction from local documents (building on the existing
`file_import.py`) through offline batch captioning of local video/audio
files, plus an automated eval harness. Live-mic captioning is explicitly
deferred but the core (`generate()` override on a stateless per-chunk
model) must not preclude it — see "Live-mic seam" below.

## Non-goals

- Live/streaming capture (future work, seam only)
- Document text extraction itself (`file_import.py` already handles this;
  term extraction consumes its output, doesn't replace it)
- Training or fine-tuning any model, including the term-extraction step —
  `pyate`'s algorithms are statistical/POS-based, not learned
- GPU/CUDA — target is Apple Silicon unified memory, MLX runtime

## Architecture

```
docs (.docx/.pdf/scans) ──file_import.py (extractous+tesseract, unmodified)──► raw text
                                                    │
                                          pyate term_extractor        [new]
                                          (POS candidates × domain-vs-
                                           general-corpus weirdness score)
                                                    │
                                          term list (.txt, one per line)
                                                    │
                                          sentencepiece(tokenizer.model)──► ContextGraphCTC   [built once]
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
| `captionlm/term_extraction.py` | doc text → `pyate.term_extractor` candidates → cleaned term list (.txt) | new |
| `captionlm/terms.py` | load term list, sentencepiece-encode, build `ContextGraphCTC` | new |
| `captionlm/biased_model.py` | `BiasedParakeetTDTCTC(ParakeetTDTCTC)`, `generate()` override | new |
| `captionlm/merge.py` | span-replacing merge of `WSHyp` into `AlignedToken` sequence | new |
| `captionlm/config.py` | model id, spotter hyperparameters (`cb_weight`, `beam_threshold`, etc.), term-extraction threshold | new |
| `captionlm/cli.py` | CLI: file/dir argument → captioned output | new |
| `captionlm/eval.py` | F-score / WER, biasing on vs off, `cb_weight` sweep flag | new |
| `captionlm/eval_dataset.py` | fetch Earnings-21/22 audio+transcript+entities, resolve ticker→SEC filing, assemble doc+audio+transcript eval triples | new |
| `captionlm/vendor/context_graph_ctc.py` | trie structure for context graph | vendored from NeMo, Apache-2.0, unmodified |
| `captionlm/vendor/ctc_word_spotter.py` | token-passing search (`run_word_spotter`) | vendored from NeMo, Apache-2.0, 2-line diff (tokenizer→vocabulary list) |
| `captionlm/vendor/context_biasing_utils.py` | `compute_fscore` (eval metric) | vendored from NeMo, Apache-2.0, unmodified — also contains `merge_alignment_with_ws_hyps`, NOT used (we use our own `merge.py` since parakeet-mlx's types differ from NeMo's `rnnt_utils.Hypothesis`) |

Each vendored file keeps a header comment with the NeMo source path and
commit/date pulled from, so it can be re-diffed on upstream changes.

### Data contracts

- **Term extraction input**: raw text from `file_import.py` (already
  produces plain strings per document). Output: plain text file, one term
  per line, UTF-8 — same contract as before, just now produced by
  `term_extraction.py` instead of handed in manually. Manual term lists in
  this exact format remain a valid input to `terms.py` directly (useful for
  the eval dataset's own entity lists, and for hand-curated overrides).
- **`term_extraction.py` algorithm**: `pyate.term_extractor` (combines
  C-value and `weirdness`) scores POS-tag-filtered candidate phrases by
  frequency-in-docs vs. frequency-in-general-corpus (pyate ships a default
  general-English reference; no corpus to build). Output kept above a
  configurable score threshold (`config.py`), deduplicated,
  lowercased-compared but original casing preserved (acronyms matter).
- **Multi-word terms**: supported. `ContextGraphCTC` builds a trie over BPE
  token sequences regardless of how many words a term spans; `merge.py`
  must therefore replace a *span* of one or more transducer words per spot,
  keyed on frame-range overlap ≥ the configured threshold (default 30%,
  matching NeMo), not a single word.
- **CLI input**: explicit file or directory argument (audio/video,
  whatever `ffmpeg`/parakeet-mlx already accepts). Fully independent of
  `file_import.py`/`term_extraction.py` and `static_assets/`, which run
  earlier in the pipeline (doc → term list) and produce the `.txt` file
  `terms.py` consumes — `cli.py` takes that term list as a separate
  argument, it doesn't re-run extraction itself.
- **Eval input**: a directory of `<clip>.wav` + `<clip>.txt` (reference
  transcript) pairs, same format whether hand-collected or produced by
  `eval_dataset.py`. `eval_dataset.py` also emits a matched `<clip>.doc.txt`
  (the SEC filing text run through term extraction) and a
  `<clip>.entities.txt` (the dataset's own ground-truth entity tags) per
  clip, so `eval.py` can score two things: ASR F-score/WER with biasing
  on/off (against `<clip>.txt`), and term-extraction precision/recall
  (extracted terms vs. `<clip>.entities.txt`) — the latter tells you
  whether `pyate` is even finding the right vocabulary, independent of
  whether the ASR half then uses it correctly.

### Automated eval dataset (Earnings-21 / Earnings-22)

[Earnings-21](https://github.com/revdotcom/speech-datasets) (44 earnings
calls, ~39h, CC-BY-SA) and Earnings-22 (125 calls, ~119h, adds ticker/HQ
metadata) ship audio + transcripts + NER-tagged entity lists via git-lfs.
Company/ticker metadata lets `eval_dataset.py` pull the matching earnings
press release (usually an 8-K Exhibit 99.1) from SEC EDGAR's free,
keyless full-text search API (`efts.sec.gov`) and company-ticker lookup
(`data.sec.gov/files/company_tickers.json`) — stdlib `urllib`, no new
dependency — giving a real doc+audio+transcript triple per call:

```
SEC EDGAR filing (doc)  ──file_import.py──► text ──term_extraction.py──► term list
                                                                              │
Earnings-21/22 audio ──────────────────────────────────────────► biased ASR ─┤
Earnings-21/22 transcript + entity tags (ground truth) ──────────────────────┴──► eval.py
```

This bootstraps `eval.py` with real, scored data instead of waiting on
hand-collected clips — start with a handful (5–10) of Earnings-21 calls
across sectors, not the full 39h.

**Ceiling, stated plainly:** ticker→filing matching won't be 100% —
foreign private issuers file 6-Ks instead of 8-Ks, some companies don't
issue a separate press release, EDGAR full-text search can mismatch on
common names. `eval_dataset.py` skips and logs unmatched calls rather than
guessing; a handful of clean matches is enough to validate the pipeline.
Hand-collected clips (the original plan) remain a fully supported fallback
input to `eval.py` and require no code change — it's the same `.wav`/`.txt`
contract either way.

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
- `pyate` + `spacy` (+ `en_core_web_sm`, ~13MB) — term extraction. MIT
  license. Archived upstream (June 2023) but functional on current
  spaCy/Python; POS-tagging only, no transformer, no GPU — noted as the
  one dependency risk in this plan (no upstream fixes if something
  breaks), accepted because rewriting a term-extraction algorithm from
  scratch is exactly the kind of wheel-reinvention this project should
  avoid.

`eval_dataset.py` adds no new dependency — SEC EDGAR access is stdlib
`urllib`/`json`; Earnings-21/22 retrieval is `git`/`git-lfs` (external
tool, not a Python dependency).

Explicitly not adding: `nemo_toolkit` (torch + CUDA-shaped, ~2GB, wrong fit
for Apple Silicon background use — this is why the spotter/graph/fscore
code is vendored instead of imported); `requests` (stdlib `urllib`
suffices for the two simple EDGAR calls needed).

## Model

`mlx-community/parakeet-tdt_ctc-110m`, set as a `config.py` constant. Only
hybrid TDT-CTC checkpoints work with this method (pure-TDT checkpoints like
`parakeet-tdt-0.6b-v2` have no CTC head). 110m chosen for footprint
(~0.25GB bf16); `1.1b` is a one-line config change if the eval harness
shows the accuracy gap matters enough to justify ~2.4GB resident.

## Testing / verification

- **Phase self-checks** (per writing-plans convention): term-extraction
  smoke test in `term_extraction.py` (known jargon-heavy paragraph scores
  above threshold, common English words don't), graph roundtrip assert in
  `terms.py`, blank-index assert in `biased_model.py`, a merge unit test
  with a synthetic single-word and a synthetic multi-word overlap case in
  `merge.py`.
- **Eval harness** (Phase 6) is the real verification, run against
  `eval_dataset.py`'s Earnings-21 triples: term-extraction precision/recall
  against the dataset's entity tags, then F-score (vendored
  `compute_fscore`) and WER (`jiwer`) with biasing on vs off, and a
  `--cb-weight` sweep flag so tuning is a rerun, not a code edit.
  Hand-collected clips remain a supported input with no code change.

## Phases

1. **Env + vendor** — add `parakeet-mlx`, `sentencepiece`, `jiwer`,
   `pyate`, `spacy` (+ `en_core_web_sm`) to venv; pull the three NeMo
   files into `vendor/`, apply the tokenizer shim diff, add source-commit
   comments.
2. **Term extraction** — `term_extraction.py` on top of `file_import.py`'s
   output: `pyate.term_extractor`, score threshold, dedup, plain-text
   output. Self-check: smoke test above.
3. **Term graph** — `terms.py` + `config.py`, consuming either
   `term_extraction.py`'s output or a hand-written list. Self-check: known
   term roundtrips through the graph.
4. **Biased model + merge** — `biased_model.py`, `merge.py`. Self-check:
   blank-index assert, merge unit test (single-word + multi-word span).
5. **CLI** — `cli.py`, file/dir argument → `.srt`/`.vtt` output using the
   biased model.
6. **Eval dataset + harness** — `eval_dataset.py` (fetch a handful of
   Earnings-21 calls, resolve ticker→SEC filing, assemble triples,
   skip/log unmatched) and `eval.py` (term-extraction precision/recall +
   ASR F-score/WER on vs off + `cb_weight` sweep).

## Open questions

None outstanding — all resolved during brainstorming (term extraction
method, eval dataset source, CLI input path, model tier, git init).
