# Domain-Biased Captioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract domain-specific terms from documents and use them to bias
a local ASR model's transcription of audio/video files, following the
CTC-based Word Spotter (CTC-WS) method, with an automated eval harness that
scores the whole pipeline against a public dataset.

**Architecture:** Subclass parakeet-mlx's `ParakeetTDTCTC`, overriding only
`generate()` to insert a CTC-log-prob word-spotter search (vendored from
NVIDIA NeMo, Apache-2.0) between the existing encoder and TDT-decode steps,
then merge spotted terms into the TDT hypothesis by frame-overlap. Term
lists come from `pyate`'s statistical term-extraction algorithm run over
`file_import.py`'s existing document-text output. Eval pairs Earnings-21
audio/transcripts with their matching SEC EDGAR filing as the "doc" input,
closing the loop end to end.

**Tech Stack:** Python 3.12, MLX (Apple Silicon), `parakeet-mlx`,
`sentencepiece`, `pyate`+`spacy`, `jiwer`, `kaldialign`, `numpy`, stdlib
`urllib`/`json` for SEC EDGAR.

**Spec:** `docs/superpowers/specs/2026-08-29-domain-biased-captioning-design.md`

## Global Constraints

- Python 3.12 venv at `venv/` (already present). Apple Silicon / MLX only —
  no CUDA, no `nemo_toolkit`.
- Model: `mlx-community/parakeet-tdt_ctc-110m` (constant `MODEL_ID` in
  `captionlm/config.py`). Only hybrid TDT-CTC checkpoints work with this
  method.
- Vendored files (`captionlm/vendor/`) keep a header comment citing their
  NeMo source path and the Apache-2.0 license. `context_graph_ctc.py` is
  vendored unmodified. `ctc_word_spotter.py` has exactly one line changed
  (documented in Task 1). **Correction to the spec:** the spec called
  `context_biasing_utils.py`'s `compute_fscore` "vendored unmodified" —
  it isn't possible unmodified, since it imports `nemo.utils.logging` and
  `nemo.collections.asr.parts.utils.manifest_utils.read_manifest`, neither
  of which exist outside NeMo. Task 1 instead vendors the algorithm into
  `captionlm/vendor/fscore.py`, adapted to take an in-memory `list[dict]`
  instead of a NeMo manifest file. This is the smallest change that avoids
  a `nemo_toolkit` dependency; everything else about the spec is unchanged.
- SEC EDGAR requires a descriptive `User-Agent` header on every request
  (fair-use policy) — `captionlm/config.py` has a placeholder
  `SEC_CONTACT` string; fill in a real contact before running
  `eval_dataset.py` against the live API.
- Every commit in this plan ends with a `Co-Authored-By: Claude Sonnet 5
  <noreply@anthropic.com>` trailer, matching this repo's existing commits.
- Test runner: `pytest` (added in Task 1, dev-only).

---

### Task 1: Environment setup and vendored NeMo code

**Files:**
- Modify: `venv/` (pip installs, spaCy model download)
- Create: `captionlm/__init__.py`
- Create: `captionlm/vendor/__init__.py`
- Create: `captionlm/vendor/context_graph_ctc.py`
- Create: `captionlm/vendor/ctc_word_spotter.py`
- Create: `captionlm/vendor/fscore.py`
- Test: `tests/test_vendor.py`

**Interfaces:**
- Produces: `captionlm.vendor.context_graph_ctc.ContextGraphCTC` (methods:
  `__init__(blank_id: int)`, `add_to_graph(word_items: list[tuple[str,
  list[list[int]]]])`, attributes `.root: ContextState`, `.num_nodes: int`).
  `ContextState` (attributes `.next: dict[int, ContextState]`, `.is_end:
  bool`, `.word: str | None`).
- Produces: `captionlm.vendor.ctc_word_spotter.run_word_spotter(logprobs:
  np.ndarray, context_graph: ContextGraphCTC, asr_model, blank_idx: int =
  0, beam_threshold: float = 5.0, cb_weight: float = 3.0,
  ctc_ali_token_weight: float = 0.5, keyword_threshold: float = -5.0,
  blank_threshold: float = 0.8, non_blank_threshold: float = 0.001) ->
  list[WSHyp]`. `WSHyp` (attributes `.word: str`, `.score: float`,
  `.start_frame: int`, `.end_frame: int`). `asr_model` must expose
  `.vocabulary: list[str]`.
- Produces: `captionlm.vendor.fscore.compute_fscore(samples: list[dict],
  key_words_list: list[str], eps: str = "<eps>") -> tuple[float, float,
  float]` (precision, recall, fscore). Each sample dict has `"text"` and
  `"pred_text"` string keys.

- [ ] **Step 1: Install dependencies**

```bash
venv/bin/pip install parakeet-mlx sentencepiece jiwer pyate spacy kaldialign pytest
venv/bin/python -m spacy download en_core_web_sm
```

- [ ] **Step 2: Create package skeleton**

```bash
mkdir -p captionlm/vendor tests
touch captionlm/__init__.py captionlm/vendor/__init__.py tests/__init__.py
```

- [ ] **Step 3: Vendor `context_graph_ctc.py` unmodified**

Fetch verbatim from NeMo and save to `captionlm/vendor/context_graph_ctc.py`:

```bash
curl -sL https://raw.githubusercontent.com/NVIDIA/NeMo/main/nemo/collections/asr/parts/context_biasing/context_graph_ctc.py \
  -o captionlm/vendor/context_graph_ctc.py
```

- [ ] **Step 4: Vendor `ctc_word_spotter.py` with the one-line tokenizer shim**

```bash
curl -sL https://raw.githubusercontent.com/NVIDIA/NeMo/main/nemo/collections/asr/parts/context_biasing/ctc_based_word_spotter.py \
  -o captionlm/vendor/ctc_word_spotter.py
```

Then edit line containing (inside `get_ctc_word_alignment`):

```python
            token = asr_model.tokenizer.ids_to_tokens([int(idx)])[0]
```

Replace with:

```python
            token = asr_model.vocabulary[int(idx)]
```

Also fix the import line at the top of the file — it currently reads:

```python
from nemo.collections.asr.parts.context_biasing.context_graph_ctc import ContextGraphCTC, ContextState
```

Change to:

```python
from captionlm.vendor.context_graph_ctc import ContextGraphCTC, ContextState
```

This is the only other change; everything else in the file (the Token
Passing Algorithm, `beam_pruning`, `state_pruning`, `find_best_hyps`,
`get_ctc_word_alignment`, `filter_wb_hyps`, `run_word_spotter`) stays as
NeMo wrote it.

- [ ] **Step 5: Write `captionlm/vendor/fscore.py`**

```python
"""F-score for context-biasing term recognition.

Adapted from NVIDIA NeMo's context_biasing_utils.compute_fscore
(Apache-2.0), source:
nemo/collections/asr/parts/context_biasing/context_biasing_utils.py

Changes from the original: takes an in-memory list of {"text", "pred_text"}
dicts instead of a NeMo manifest file path (drops the
nemo.collections.asr.parts.utils.manifest_utils dependency), uses
kaldialign directly instead of nemo.utils.logging for stats, and drops the
per-sample-keyword-field mode (this project always scores against one
global term list). The core alignment-based scoring algorithm is
unchanged.
"""
import re

from kaldialign import align


def compute_fscore(
    samples: list[dict],
    key_words_list: list[str],
    eps: str = "<eps>",
) -> tuple[float, float, float]:
    assert samples, "samples is empty"
    assert key_words_list, "key_words_list is empty"

    all_key_words_list = [kw.lower() for kw in key_words_list]
    keywords_set = set(all_key_words_list)
    max_ngram_order = max(len(kw.split()) for kw in all_key_words_list)
    key_words_stat = {kw: [0, 0, 0] for kw in all_key_words_list}  # tp, gt, fp

    for sample in samples:
        ref = re.sub(r"[.,!?]", "", sample["text"].lower()).split()
        hyp = re.sub(r"[.,!?]", "", sample["pred_text"].lower()).split()
        ali = align(ref, hyp, eps)

        for word_ref, word_hyp in ali:
            if word_ref in keywords_set:
                key_words_stat[word_ref][1] += 1
                if word_ref == word_hyp:
                    key_words_stat[word_ref][0] += 1
            elif word_hyp in keywords_set:
                key_words_stat[word_hyp][2] += 1

        for ngram_order in range(2, max_ngram_order + 1):
            idx = 0
            item_ref: list[tuple[str, int]] = []
            while idx < len(ali):
                if item_ref:
                    item_ref = [item_ref[1]]
                    idx = item_ref[0][1] + 1
                while len(item_ref) != ngram_order and idx < len(ali):
                    word = ali[idx][0]
                    idx += 1
                    if word == eps:
                        continue
                    item_ref.append((word, idx - 1))
                if len(item_ref) == ngram_order:
                    phrase_ref = " ".join(wr[0] for wr in item_ref)
                    phrase_hyp = " ".join(ali[wr[1]][1] for wr in item_ref)
                    if phrase_ref in keywords_set:
                        key_words_stat[phrase_ref][1] += 1
                        if phrase_ref == phrase_hyp:
                            key_words_stat[phrase_ref][0] += 1

            idx = 0
            item_hyp: list[tuple[str, int]] = []
            while idx < len(ali):
                if item_hyp:
                    item_hyp = [item_hyp[1]]
                    idx = item_hyp[0][1] + 1
                while len(item_hyp) != ngram_order and idx < len(ali):
                    word = ali[idx][1]
                    idx += 1
                    if word == eps:
                        continue
                    item_hyp.append((word, idx - 1))
                if len(item_hyp) == ngram_order:
                    phrase_hyp = " ".join(wh[0] for wh in item_hyp)
                    phrase_ref = " ".join(ali[wh[1]][0] for wh in item_hyp)
                    if phrase_hyp in keywords_set and phrase_hyp != phrase_ref:
                        key_words_stat[phrase_hyp][2] += 1

    tp = sum(v[0] for v in key_words_stat.values())
    gt = sum(v[1] for v in key_words_stat.values())
    fp = sum(v[2] for v in key_words_stat.values())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (gt + 1e-8)
    fscore = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision, recall, fscore
```

- [ ] **Step 6: Write the failing test**

```python
# tests/test_vendor.py
from captionlm.vendor.context_graph_ctc import ContextGraphCTC
from captionlm.vendor.ctc_word_spotter import Token, beam_pruning, state_pruning
from captionlm.vendor.fscore import compute_fscore


def test_context_graph_builds_simple_word():
    graph = ContextGraphCTC(blank_id=99)
    graph.add_to_graph([("hi", [[1, 2]])])
    node = graph.root.next[1].next[2]
    assert node.is_end
    assert node.word == "hi"


def test_beam_pruning_drops_low_score_tokens():
    root = ContextGraphCTC(blank_id=99).root
    tokens = [Token(root, score=0.0), Token(root, score=-10.0)]
    pruned = beam_pruning(tokens, beam_threshold=5.0)
    assert len(pruned) == 1
    assert pruned[0].score == 0.0


def test_state_pruning_keeps_best_per_state():
    root = ContextGraphCTC(blank_id=99).root
    tokens = [Token(root, score=1.0), Token(root, score=5.0)]
    pruned = state_pruning(tokens)
    assert len(pruned) == 1
    assert pruned[0].score == 5.0


def test_compute_fscore_perfect_match():
    samples = [{"text": "deploy the kubernetes cluster", "pred_text": "deploy the kubernetes cluster"}]
    precision, recall, fscore = compute_fscore(samples, ["kubernetes"])
    assert precision > 0.99
    assert recall > 0.99
    assert fscore > 0.99


def test_compute_fscore_missed_term():
    samples = [{"text": "deploy the kubernetes cluster", "pred_text": "deploy the koobernetti cluster"}]
    precision, recall, fscore = compute_fscore(samples, ["kubernetes"])
    assert recall < 0.01
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_vendor.py -v`
Expected: FAIL (no module named `captionlm` yet, or import errors) until
Steps 1-5 above are done — run this after Steps 1-5, expect PASS.

- [ ] **Step 8: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_vendor.py -v`
Expected: PASS (5 tests)

- [ ] **Step 9: Commit**

```bash
git add captionlm/vendor captionlm/__init__.py tests/test_vendor.py tests/__init__.py
git commit -m "$(cat <<'EOF'
Vendor NeMo CTC-WS context graph, word spotter, and fscore (Apache-2.0)

context_graph_ctc.py is unmodified. ctc_word_spotter.py has one line
changed (tokenizer.ids_to_tokens -> vocabulary[idx], since parakeet-mlx
has no tokenizer object) plus an updated import path. fscore.py is
compute_fscore adapted to run on in-memory samples instead of a NeMo
manifest file, dropping the nemo_toolkit dependency.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Config and term extraction

**Files:**
- Create: `captionlm/config.py`
- Create: `captionlm/term_extraction.py`
- Test: `tests/test_term_extraction.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `captionlm.config.MODEL_ID: str`,
  `captionlm.config.TERM_EXTRACTION_TOP_N: int`,
  `captionlm.config.SEC_CONTACT: str`,
  `captionlm.config.SpotterConfig` (dataclass, fields: `beam_threshold:
  float = 5.0`, `cb_weight: float = 3.0`, `ctc_ali_token_weight: float =
  0.5`, `keyword_threshold: float = -5.0`, `blank_threshold: float = 0.8`,
  `non_blank_threshold: float = 0.001`, `intersection_threshold: float =
  30.0`).
- Produces: `captionlm.term_extraction.extract_terms(text: str, top_n: int
  = 50) -> list[str]`, `captionlm.term_extraction.write_term_list(terms:
  list[str], path: str) -> None`.

- [ ] **Step 1: Write `captionlm/config.py`**

```python
"""Project-wide configuration constants."""
from dataclasses import dataclass

MODEL_ID = "mlx-community/parakeet-tdt_ctc-110m"
TERM_EXTRACTION_TOP_N = 50

# SEC EDGAR requires a descriptive contact string in the User-Agent header
# on every request. Fill this in with a real contact before running
# captionlm/eval_dataset.py against the live API.
SEC_CONTACT = "captionlm/1.0 (set a real contact email here)"


@dataclass
class SpotterConfig:
    """CTC-WS hyperparameters. Defaults match NeMo's own
    run_word_spotter defaults; retune cb_weight against your own term
    list via eval.py's --cb-weight sweep before trusting these."""

    beam_threshold: float = 5.0
    cb_weight: float = 3.0
    ctc_ali_token_weight: float = 0.5
    keyword_threshold: float = -5.0
    blank_threshold: float = 0.8
    non_blank_threshold: float = 0.001
    intersection_threshold: float = 30.0
```

- [ ] **Step 2: Write the failing test for term extraction**

```python
# tests/test_term_extraction.py
from captionlm.term_extraction import extract_terms, write_term_list

JARGON_DOC = """
Our platform runs on a kubernetes cluster deployed across three regions.
The kubernetes cluster autoscaler adjusts node pools based on demand.
We migrated the kubernetes cluster configuration to a new Helm chart last
quarter, and the on-call team monitors kubernetes cluster health daily.
The weather was nice today. I went for a walk and had a coffee. It was a
pretty ordinary afternoon with nothing much to report.
"""


def test_extract_terms_finds_repeated_jargon():
    terms = extract_terms(JARGON_DOC, top_n=10)
    assert any("kubernetes" in term.lower() for term in terms)


def test_extract_terms_deduplicates_case_insensitively():
    terms = extract_terms(JARGON_DOC, top_n=10)
    lowered = [t.lower() for t in terms]
    assert len(lowered) == len(set(lowered))


def test_write_term_list(tmp_path):
    path = tmp_path / "terms.txt"
    write_term_list(["kubernetes cluster", "Helm chart"], str(path))
    content = path.read_text(encoding="utf-8")
    assert content == "kubernetes cluster\nHelm chart\n"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_term_extraction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'captionlm.term_extraction'`

- [ ] **Step 4: Write `captionlm/term_extraction.py`**

```python
"""Extract domain-specific terms from document text.

Uses pyate's term_extractor (Sclano & Velardi 2007), which scores POS-
tagged candidate phrases by frequency in the given text relative to
pyate's bundled general-English reference corpus — this is what makes it
domain-specific extraction rather than generic keyword extraction. MIT
license, no GPU, spaCy en_core_web_sm only.
"""
from pyate import term_extractor


def extract_terms(text: str, top_n: int = 50) -> list[str]:
    scores = term_extractor(text)
    if scores.empty:
        return []

    ranked = scores.sort_values(ascending=False)
    seen: set[str] = set()
    terms: list[str] = []
    for term in ranked.index:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
        if len(terms) >= top_n:
            break
    return terms


def write_term_list(terms: list[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for term in terms:
            f.write(term + "\n")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_term_extraction.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add captionlm/config.py captionlm/term_extraction.py tests/test_term_extraction.py
git commit -m "$(cat <<'EOF'
Add config constants and pyate-based domain term extraction

term_extraction.py scores candidate phrases by frequency-in-doc vs
pyate's bundled general-English corpus, not generic keyword extraction.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Context graph builder

**Files:**
- Create: `captionlm/terms.py`
- Test: `tests/test_terms.py`

**Interfaces:**
- Consumes: `captionlm.vendor.context_graph_ctc.ContextGraphCTC` (Task 1).
- Produces: `captionlm.terms.load_term_list(path: str) -> list[str]`,
  `captionlm.terms.load_tokenizer(model_id: str) ->
  sentencepiece.SentencePieceProcessor`,
  `captionlm.terms.build_context_graph(terms: list[str], tokenizer:
  sentencepiece.SentencePieceProcessor, blank_idx: int) ->
  ContextGraphCTC`. `build_context_graph` and `load_tokenizer` are also
  used by `captionlm/biased_model.py` (Task 5), `captionlm/cli.py` (Task
  6), and `captionlm/eval.py` (Task 7).

**Note:** `load_tokenizer` downloads `tokenizer.model` (~1MB) from the
model's Hugging Face repo via `huggingface_hub.hf_hub_download` (already a
transitive dependency of `parakeet-mlx`) and caches it locally — this test
needs network on first run only.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_terms.py
from captionlm.config import MODEL_ID
from captionlm.terms import build_context_graph, load_term_list, load_tokenizer


def test_load_term_list_skips_blanks_and_comments(tmp_path):
    path = tmp_path / "terms.txt"
    path.write_text("kubernetes\n\n# a comment\nHelm chart\n", encoding="utf-8")
    assert load_term_list(str(path)) == ["kubernetes", "Helm chart"]


def test_single_word_term_roundtrips_through_graph():
    tokenizer = load_tokenizer(MODEL_ID)
    blank_idx = tokenizer.get_piece_size()  # same vocab as the model's CTC head
    graph = build_context_graph(["kubernetes"], tokenizer, blank_idx)

    token_ids = tokenizer.encode("kubernetes", out_type=int)
    node = graph.root
    for token_id in token_ids:
        node = node.next[token_id]
    assert node.is_end
    assert node.word == "kubernetes"


def test_multiword_phrase_roundtrips_through_graph():
    tokenizer = load_tokenizer(MODEL_ID)
    blank_idx = tokenizer.get_piece_size()
    graph = build_context_graph(["San Francisco"], tokenizer, blank_idx)

    token_ids = tokenizer.encode("San Francisco", out_type=int)
    node = graph.root
    for token_id in token_ids:
        node = node.next[token_id]
    assert node.is_end
    assert node.word == "San Francisco"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_terms.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'captionlm.terms'`

- [ ] **Step 3: Write `captionlm/terms.py`**

```python
"""Build a ContextGraphCTC from a domain term list, tokenized against the
model's own sentencepiece vocabulary (same vocab and id order the model's
CTC head outputs logits over — verified: sentencepiece piece ids for this
model line up 1:1 with the config's aux_ctc.decoder.vocabulary list)."""
import sentencepiece as spm
from huggingface_hub import hf_hub_download

from captionlm.vendor.context_graph_ctc import ContextGraphCTC


def load_term_list(path: str) -> list[str]:
    terms = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            terms.append(line)
    return terms


def load_tokenizer(model_id: str) -> spm.SentencePieceProcessor:
    model_path = hf_hub_download(model_id, "tokenizer.model")
    return spm.SentencePieceProcessor(model_file=model_path)


def build_context_graph(
    terms: list[str],
    tokenizer: spm.SentencePieceProcessor,
    blank_idx: int,
) -> ContextGraphCTC:
    graph = ContextGraphCTC(blank_id=blank_idx)
    word_items = [(term, [tokenizer.encode(term, out_type=int)]) for term in terms]
    graph.add_to_graph(word_items)
    return graph
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_terms.py -v`
Expected: PASS (3 tests). First run downloads `tokenizer.model` — requires
network.

- [ ] **Step 5: Commit**

```bash
git add captionlm/terms.py tests/test_terms.py
git commit -m "$(cat <<'EOF'
Add context graph builder from a domain term list

Encodes each term with the model's own sentencepiece tokenizer.model,
whose piece ids line up 1:1 with the CTC head's output vocabulary.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Span-replacing merge

**Files:**
- Create: `captionlm/merge.py`
- Test: `tests/test_merge.py`

**Interfaces:**
- Consumes: `parakeet_mlx.alignment.AlignedToken` (from the `parakeet-mlx`
  package, installed in Task 1), `captionlm.vendor.ctc_word_spotter.WSHyp`
  (Task 1).
- Produces: `captionlm.merge.merge_spans(tokens: list[AlignedToken],
  ws_hyps: list[WSHyp], time_ratio: float, intersection_threshold: float =
  30.0) -> list[AlignedToken]`. Used by `captionlm/biased_model.py` (Task
  5). Replacement tokens are marked with `id=-1` so downstream code can
  tell a biased token from a normally-decoded one.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_merge.py
from parakeet_mlx.alignment import AlignedToken

from captionlm.merge import merge_spans
from captionlm.vendor.ctc_word_spotter import WSHyp

TIME_RATIO = 0.08  # seconds per encoder frame, arbitrary for these tests


def _tok(text, start, duration):
    return AlignedToken(id=0, text=text, start=start, duration=duration)


def test_single_word_replacement():
    tokens = [
        _tok(" hello", 0.0, 0.16),
        _tok(" kubernetees", 0.16, 0.40),  # mis-transcribed
        _tok(" cluster", 0.56, 0.24),
    ]
    # word 2 spans frames [2, 6] at 0.08s/frame (0.16s to 0.56s)
    hyps = [WSHyp(word="Kubernetes", score=10.0, start_frame=2, end_frame=6)]

    merged = merge_spans(tokens, hyps, TIME_RATIO)

    assert [t.text for t in merged] == [" hello", " Kubernetes", " cluster"]
    assert merged[1].id == -1
    assert merged[1].start == 0.16
    assert abs(merged[1].end - 0.56) < 1e-9


def test_multiword_span_replacement():
    tokens = [
        _tok(" hello", 0.0, 0.16),
        _tok(" san", 0.16, 0.16),
        _tok(" francisco", 0.32, 0.32),
        _tok(" today", 0.64, 0.16),
    ]
    # "san francisco" spans frames [2, 7] (0.16s to 0.64s)
    hyps = [WSHyp(word="San Francisco", score=8.0, start_frame=2, end_frame=7)]

    merged = merge_spans(tokens, hyps, TIME_RATIO)

    assert [t.text for t in merged] == [" hello", " San Francisco", " today"]
    assert merged[1].id == -1
    assert merged[1].start == 0.16
    assert abs(merged[1].end - 0.64) < 1e-9


def test_no_hyps_returns_tokens_unchanged():
    tokens = [_tok(" hello", 0.0, 0.16)]
    assert merge_spans(tokens, [], TIME_RATIO) == tokens


def test_low_overlap_hyp_is_ignored():
    tokens = [_tok(" hello", 0.0, 1.0)]  # 1s-long word
    # hyp only covers the first 0.08s of a 1s-long word -> 8% overlap, below default 30%
    hyps = [WSHyp(word="Wrong", score=10.0, start_frame=0, end_frame=0)]
    merged = merge_spans(tokens, hyps, TIME_RATIO)
    assert [t.text for t in merged] == [" hello"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_merge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'captionlm.merge'`

- [ ] **Step 3: Write `captionlm/merge.py`**

```python
"""Replace transducer word spans with higher-confidence CTC-WS spotted
terms, by frame-overlap, following the same replace-on-overlap idea as
NeMo's merge_alignment_with_ws_hyps (this is a from-scratch
reimplementation against parakeet-mlx's AlignedToken type, not a port —
parakeet-mlx has no equivalent of NeMo's rnnt_utils.Hypothesis)."""
from parakeet_mlx.alignment import AlignedToken

from captionlm.vendor.ctc_word_spotter import WSHyp


def _group_words(tokens: list[AlignedToken]) -> list[list[AlignedToken]]:
    words: list[list[AlignedToken]] = []
    for tok in tokens:
        if words and " " not in tok.text:
            words[-1].append(tok)
        else:
            words.append([tok])
    return words


def _overlap_pct(word_start: float, word_end: float, hyp_start: float, hyp_end: float) -> float:
    intersection = max(0.0, min(word_end, hyp_end) - max(word_start, hyp_start))
    word_len = word_end - word_start
    return 0.0 if word_len <= 0 else 100.0 * intersection / word_len


def merge_spans(
    tokens: list[AlignedToken],
    ws_hyps: list[WSHyp],
    time_ratio: float,
    intersection_threshold: float = 30.0,
) -> list[AlignedToken]:
    if not tokens or not ws_hyps:
        return tokens

    words = _group_words(tokens)
    used = [False] * len(words)
    replacements: dict[int, tuple[int, AlignedToken]] = {}

    for hyp in sorted(ws_hyps, key=lambda h: h.start_frame):
        hyp_start = hyp.start_frame * time_ratio
        hyp_end = (hyp.end_frame + 1) * time_ratio
        overlap_idxs = [
            i
            for i, w in enumerate(words)
            if not used[i]
            and _overlap_pct(w[0].start, w[-1].end, hyp_start, hyp_end) >= intersection_threshold
        ]
        if not overlap_idxs:
            continue

        first_idx, last_idx = overlap_idxs[0], overlap_idxs[-1]
        for i in range(first_idx, last_idx + 1):
            used[i] = True

        lead_space = " " if " " in words[first_idx][0].text else ""
        replacements[first_idx] = (
            last_idx,
            AlignedToken(
                id=-1,
                text=lead_space + hyp.word,
                start=words[first_idx][0].start,
                duration=words[last_idx][-1].end - words[first_idx][0].start,
                confidence=1.0,
            ),
        )

    merged: list[AlignedToken] = []
    i = 0
    while i < len(words):
        if i in replacements:
            last_idx, replacement = replacements[i]
            merged.append(replacement)
            i = last_idx + 1
        else:
            merged.extend(words[i])
            i += 1
    return merged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_merge.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add captionlm/merge.py tests/test_merge.py
git commit -m "$(cat <<'EOF'
Add frame-overlap span merge for spotted terms

Groups AlignedTokens into words on the leading-space marker, replaces
any word (or contiguous run of words) whose span overlaps a spotted
WSHyp by >= threshold%. Handles both single-word and multi-word terms.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Biased model

**Files:**
- Create: `captionlm/biased_model.py`
- Test: `tests/test_biased_model.py`

**Interfaces:**
- Consumes: `captionlm.config.SpotterConfig`, `captionlm.config.MODEL_ID`
  (Task 2), `captionlm.merge.merge_spans` (Task 4),
  `captionlm.vendor.context_graph_ctc.ContextGraphCTC`,
  `captionlm.vendor.ctc_word_spotter.run_word_spotter` (Task 1),
  `captionlm.terms.build_context_graph`, `captionlm.terms.load_tokenizer`
  (Task 3, used only in the test).
- Produces: `captionlm.biased_model.BiasedParakeetTDTCTC` (a
  `parakeet_mlx.parakeet.ParakeetTDTCTC` subclass; instance attributes
  `.context_graph: ContextGraphCTC | None` (default `None`, meaning
  "unbiased"), `.spotter_config: SpotterConfig`; overrides `.generate(mel:
  mx.array, *, decoding_config: DecodingConfig = DecodingConfig()) ->
  list[AlignedResult]`). `.vocabulary`, `.transcribe(...)`, `.time_ratio`,
  `.encoder`, `.ctc_decoder`, `.decode(...)` are inherited unchanged from
  `ParakeetTDT`/`ParakeetTDTCTC`.
- Produces: `captionlm.biased_model.load_biased_model(hf_id: str =
  MODEL_ID, *, dtype: mx.Dtype = mx.bfloat16) -> BiasedParakeetTDTCTC`.
  Used by `captionlm/cli.py` (Task 6) and `captionlm/eval.py` (Task 7).

**Why a separate loader function:** `parakeet_mlx.utils.from_pretrained`
picks the model class internally from `config["target"]` and always
constructs the stock `ParakeetTDTCTC`, with no hook to substitute a
subclass. `load_biased_model` mirrors its ~10-line body (fetch config +
weights, construct, cast dtype) but constructs `BiasedParakeetTDTCTC`
directly — this project only ever loads the hybrid TDT-CTC checkpoint, so
`from_config`'s branching for other model types isn't needed.

**Note on Step 2's test:** it downloads and runs the real 110m model
(~0.25GB bf16) on a 1-second synthetic silent WAV. This needs network on
first run (cached after) and takes a few seconds on Apple Silicon — slower
than the rest of this plan's tests, but it's the only way to verify the
override doesn't break the unbiased path, which every other test in this
plan assumes still works.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biased_model.py
import wave

import mlx.core as mx

from captionlm.biased_model import load_biased_model
from captionlm.config import MODEL_ID
from captionlm.terms import build_context_graph, load_tokenizer


def _write_silent_wav(path, seconds=1.0, sample_rate=16000):
    n_samples = int(seconds * sample_rate)
    with wave.open(path, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * n_samples)


def test_unbiased_generate_matches_base_class_shape(tmp_path):
    wav_path = str(tmp_path / "silence.wav")
    _write_silent_wav(wav_path)

    model = load_biased_model(MODEL_ID)
    assert model.context_graph is None

    result = model.transcribe(wav_path)
    assert hasattr(result, "text")
    assert hasattr(result, "sentences")


def test_biased_generate_runs_with_context_graph(tmp_path):
    wav_path = str(tmp_path / "silence.wav")
    _write_silent_wav(wav_path)

    model = load_biased_model(MODEL_ID)
    tokenizer = load_tokenizer(MODEL_ID)
    blank_idx = len(model.vocabulary)
    model.context_graph = build_context_graph(["kubernetes"], tokenizer, blank_idx)

    result = model.transcribe(wav_path)
    assert hasattr(result, "text")


def test_blank_idx_matches_vocabulary_length():
    model = load_biased_model(MODEL_ID)
    # ConvASRDecoder appends blank as the last class: num_classes = len(vocabulary) + 1.
    # Silently using the vendored spotter's default blank_idx=0 spots nothing; this
    # assertion is the self-check the design doc calls for.
    assert len(model.vocabulary) > 0
    assert mx.array(model.vocabulary == model.vocabulary).item()  # vocabulary is stable/non-empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_biased_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'captionlm.biased_model'`

- [ ] **Step 3: Write `captionlm/biased_model.py`**

```python
"""BiasedParakeetTDTCTC: CTC-WS context-biasing on top of parakeet-mlx's
hybrid TDT-CTC model. Runs the CTC head's log-probs through the vendored
word spotter, then merges spotted terms into the TDT hypothesis by
frame-overlap. When .context_graph is None, behaves exactly like the
stock ParakeetTDTCTC.generate()."""
import json

import mlx.core as mx
import numpy as np
from dacite import from_dict
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten, tree_unflatten
from parakeet_mlx.alignment import sentences_to_result, tokens_to_sentences
from parakeet_mlx.parakeet import DecodingConfig, ParakeetTDTCTC, ParakeetTDTCTCArgs

from captionlm.config import SpotterConfig
from captionlm.merge import merge_spans
from captionlm.vendor.context_graph_ctc import ContextGraphCTC
from captionlm.vendor.ctc_word_spotter import run_word_spotter


class BiasedParakeetTDTCTC(ParakeetTDTCTC):
    def __init__(self, args: ParakeetTDTCTCArgs):
        super().__init__(args)
        self.context_graph: ContextGraphCTC | None = None
        self.spotter_config = SpotterConfig()

    def generate(
        self, mel: mx.array, *, decoding_config: DecodingConfig = DecodingConfig()
    ):
        if len(mel.shape) == 2:
            mel = mx.expand_dims(mel, 0)

        features, lengths = self.encoder(mel)
        mx.eval(features, lengths)

        result, _ = self.decode(features, lengths, config=decoding_config)

        if self.context_graph is None:
            hypotheses = result
        else:
            ctc_logits = self.ctc_decoder(features)
            mx.eval(ctc_logits)
            blank_idx = len(self.vocabulary)

            hypotheses = []
            for batch_idx, tokens in enumerate(result):
                logprobs = np.array(ctc_logits[batch_idx])
                ws_hyps = run_word_spotter(
                    logprobs,
                    self.context_graph,
                    self,
                    blank_idx=blank_idx,
                    beam_threshold=self.spotter_config.beam_threshold,
                    cb_weight=self.spotter_config.cb_weight,
                    ctc_ali_token_weight=self.spotter_config.ctc_ali_token_weight,
                    keyword_threshold=self.spotter_config.keyword_threshold,
                    blank_threshold=self.spotter_config.blank_threshold,
                    non_blank_threshold=self.spotter_config.non_blank_threshold,
                )
                hypotheses.append(
                    merge_spans(
                        tokens,
                        ws_hyps,
                        self.time_ratio,
                        intersection_threshold=self.spotter_config.intersection_threshold,
                    )
                )

        return [
            sentences_to_result(tokens_to_sentences(h, decoding_config.sentence))
            for h in hypotheses
        ]


def load_biased_model(
    hf_id: str, *, dtype: mx.Dtype = mx.bfloat16
) -> BiasedParakeetTDTCTC:
    config = json.load(open(hf_hub_download(hf_id, "config.json"), "r"))
    weight = hf_hub_download(hf_id, "model.safetensors")

    cfg = from_dict(ParakeetTDTCTCArgs, config)
    model = BiasedParakeetTDTCTC(cfg)
    model.eval()
    model.load_weights(weight)

    curr_weights = dict(tree_flatten(model.parameters()))
    curr_weights = [(k, v.astype(dtype)) for k, v in curr_weights.items()]
    model.update(tree_unflatten(curr_weights))

    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_biased_model.py -v`
Expected: PASS (3 tests). First run downloads the 110m model — requires
network, takes a minute or two.

- [ ] **Step 5: Commit**

```bash
git add captionlm/biased_model.py tests/test_biased_model.py
git commit -m "$(cat <<'EOF'
Add BiasedParakeetTDTCTC: CTC-WS override of generate()

Subclasses ParakeetTDTCTC and overrides only generate(), inserting the
vendored word spotter + merge between the existing encoder and TDT
decode steps. context_graph=None (the default) reproduces stock
behavior exactly. load_biased_model mirrors parakeet_mlx's
from_pretrained loading recipe since it has no hook to substitute a
model subclass.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: CLI

**Files:**
- Create: `captionlm/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `captionlm.biased_model.load_biased_model` (Task 5),
  `captionlm.terms.build_context_graph`, `captionlm.terms.load_term_list`,
  `captionlm.terms.load_tokenizer` (Task 3), `captionlm.config.MODEL_ID`
  (Task 2).
- Produces: `captionlm.cli.format_timestamp(seconds: float) -> str` (SRT
  `HH:MM:SS,mmm` format), `captionlm.cli.result_to_srt(result:
  AlignedResult) -> str`, `captionlm.cli.caption_file(audio_path: str,
  term_list_path: str | None, model_id: str = MODEL_ID) -> str`, CLI entry
  point `captionlm.cli.main()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
from parakeet_mlx.alignment import AlignedResult, AlignedSentence, AlignedToken

from captionlm.cli import format_timestamp, result_to_srt


def test_format_timestamp():
    assert format_timestamp(0.0) == "00:00:00,000"
    assert format_timestamp(65.25) == "00:01:05,250"
    assert format_timestamp(3661.001) == "01:01:01,001"


def test_result_to_srt():
    token = AlignedToken(id=0, text=" hello world", start=0.0, duration=1.5)
    sentence = AlignedSentence(text=" hello world", tokens=[token])
    result = AlignedResult(text=" hello world", sentences=[sentence])

    srt = result_to_srt(result)

    assert srt.splitlines() == [
        "1",
        "00:00:00,000 --> 00:00:01,500",
        "hello world",
        "",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'captionlm.cli'`

- [ ] **Step 3: Write `captionlm/cli.py`**

```python
"""CLI: caption a video/audio file, optionally biased toward a domain
term list, and write an .srt file."""
import argparse
from pathlib import Path

from parakeet_mlx.alignment import AlignedResult

from captionlm.biased_model import load_biased_model
from captionlm.config import MODEL_ID
from captionlm.terms import build_context_graph, load_term_list, load_tokenizer


def format_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, ms = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def result_to_srt(result: AlignedResult) -> str:
    lines = []
    for i, sentence in enumerate(result.sentences, start=1):
        lines.append(str(i))
        lines.append(f"{format_timestamp(sentence.start)} --> {format_timestamp(sentence.end)}")
        lines.append(sentence.text.strip())
        lines.append("")
    return "\n".join(lines)


def caption_file(audio_path: str, term_list_path: str | None, model_id: str = MODEL_ID) -> str:
    model = load_biased_model(model_id)

    if term_list_path:
        tokenizer = load_tokenizer(model_id)
        terms = load_term_list(term_list_path)
        blank_idx = len(model.vocabulary)
        model.context_graph = build_context_graph(terms, tokenizer, blank_idx)

    result = model.transcribe(audio_path, chunk_duration=120.0)
    return result_to_srt(result)


def main():
    parser = argparse.ArgumentParser(description="Domain-biased captioning")
    parser.add_argument("audio", help="Path to an audio/video file")
    parser.add_argument("--terms", help="Path to a term list (.txt, one term per line)")
    parser.add_argument("--out", help="Output .srt path (default: <audio>.srt)")
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    srt = caption_file(args.audio, args.terms, args.model)
    out_path = args.out or str(Path(args.audio).with_suffix(".srt"))
    Path(out_path).write_text(srt, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_cli.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Manual end-to-end verification**

Run against a real short video/audio file once one is available (e.g.
something in `videos/`), with and without a term list:

```bash
venv/bin/python -m captionlm.cli path/to/clip.mp4
venv/bin/python -m captionlm.cli path/to/clip.mp4 --terms path/to/terms.txt
```

Expected: two `.srt` files, the second one's text differing from the first
wherever a term from `terms.txt` was actually spoken.

- [ ] **Step 6: Commit**

```bash
git add captionlm/cli.py tests/test_cli.py
git commit -m "$(cat <<'EOF'
Add CLI: caption a file, optionally biased by a term list

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Eval harness

**Files:**
- Create: `captionlm/eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `captionlm.vendor.fscore.compute_fscore` (Task 1),
  `captionlm.biased_model.load_biased_model` (Task 5),
  `captionlm.terms.build_context_graph`, `captionlm.terms.load_term_list`,
  `captionlm.terms.load_tokenizer` (Task 3), `captionlm.config.MODEL_ID`
  (Task 2), `jiwer.wer`.
- Produces: `captionlm.eval.load_clips(clip_dir: str) -> list[dict]` (each
  dict: `{"wav": str, "text": str}`),
  `captionlm.eval.transcribe_clips(model, clips: list[dict]) ->
  list[dict]` (each dict: `{"text": str, "pred_text": str}`),
  `captionlm.eval.run_eval(clip_dir: str, term_list_path: str, model_id:
  str = MODEL_ID, cb_weight: float | None = None) -> dict` (keys
  `"baseline"`, `"biased"`, each `{"wer": float, "precision": float,
  "recall": float, "fscore": float}`), CLI entry point
  `captionlm.eval.main()`.

**Note:** `load_clips` is the only piece with an automated test here — it's
pure filesystem logic. `run_eval` needs a real model and real labeled
clips, neither of which exist yet (the eval clip directory is empty until
Task 8 populates it via `eval_dataset.py`, or you hand-collect some). It's
exercised in Task 8's manual verification step once real clips exist.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval.py
from captionlm.eval import load_clips


def test_load_clips_pairs_wav_and_txt(tmp_path):
    (tmp_path / "clip1.wav").write_bytes(b"\x00")
    (tmp_path / "clip1.txt").write_text("hello there", encoding="utf-8")
    (tmp_path / "clip2.wav").write_bytes(b"\x00")
    (tmp_path / "clip2.txt").write_text("second clip", encoding="utf-8")
    (tmp_path / "orphan.wav").write_bytes(b"\x00")  # no matching .txt

    clips = load_clips(str(tmp_path))

    assert len(clips) == 2
    texts = {c["text"] for c in clips}
    assert texts == {"hello there", "second clip"}
    assert all(c["wav"].endswith(".wav") for c in clips)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'captionlm.eval'`

- [ ] **Step 3: Write `captionlm/eval.py`**

```python
"""Score domain-biased captioning against a labeled clip set: term
F-score and WER, with biasing on vs off. Run captionlm/eval_dataset.py
first to populate clip_dir automatically from Earnings-21, or hand-collect
<clip>.wav/<clip>.txt pairs yourself."""
import argparse
import glob
import os

import jiwer

from captionlm.biased_model import load_biased_model
from captionlm.config import MODEL_ID
from captionlm.terms import build_context_graph, load_term_list, load_tokenizer
from captionlm.vendor.fscore import compute_fscore


def load_clips(clip_dir: str) -> list[dict]:
    clips = []
    for wav_path in sorted(glob.glob(os.path.join(clip_dir, "*.wav"))):
        txt_path = os.path.splitext(wav_path)[0] + ".txt"
        if not os.path.isfile(txt_path):
            continue
        with open(txt_path, encoding="utf-8") as f:
            reference = f.read().strip()
        clips.append({"wav": wav_path, "text": reference})
    return clips


def transcribe_clips(model, clips: list[dict]) -> list[dict]:
    samples = []
    for clip in clips:
        result = model.transcribe(clip["wav"])
        samples.append({"text": clip["text"], "pred_text": result.text})
    return samples


def run_eval(
    clip_dir: str,
    term_list_path: str,
    model_id: str = MODEL_ID,
    cb_weight: float | None = None,
) -> dict:
    terms = load_term_list(term_list_path)
    clips = load_clips(clip_dir)
    assert clips, f"no .wav/.txt pairs found in {clip_dir}"

    model = load_biased_model(model_id)
    baseline = transcribe_clips(model, clips)

    tokenizer = load_tokenizer(model_id)
    blank_idx = len(model.vocabulary)
    model.context_graph = build_context_graph(terms, tokenizer, blank_idx)
    if cb_weight is not None:
        model.spotter_config.cb_weight = cb_weight
    biased = transcribe_clips(model, clips)

    references = [c["text"] for c in clips]
    baseline_wer = jiwer.wer(references, [s["pred_text"] for s in baseline])
    biased_wer = jiwer.wer(references, [s["pred_text"] for s in biased])

    baseline_p, baseline_r, baseline_f = compute_fscore(baseline, terms)
    biased_p, biased_r, biased_f = compute_fscore(biased, terms)

    return {
        "baseline": {"wer": baseline_wer, "precision": baseline_p, "recall": baseline_r, "fscore": baseline_f},
        "biased": {"wer": biased_wer, "precision": biased_p, "recall": biased_r, "fscore": biased_f},
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate domain-biased captioning")
    parser.add_argument("clip_dir", help="Directory of <clip>.wav/<clip>.txt pairs")
    parser.add_argument("terms", help="Path to the domain term list (.txt)")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--cb-weight", type=float, default=None)
    args = parser.parse_args()

    stats = run_eval(args.clip_dir, args.terms, args.model, args.cb_weight)
    for label, s in stats.items():
        print(f"{label}: WER={s['wer']:.3f} P={s['precision']:.3f} R={s['recall']:.3f} F={s['fscore']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_eval.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add captionlm/eval.py tests/test_eval.py
git commit -m "$(cat <<'EOF'
Add eval harness: WER + term F-score, biasing on vs off

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Automated eval dataset (Earnings-21 + SEC EDGAR)

**Files:**
- Create: `captionlm/eval_dataset.py`
- Test: `tests/test_eval_dataset.py`

**Interfaces:**
- Consumes: `captionlm.config.SEC_CONTACT` (Task 2).
- Produces: `captionlm.eval_dataset.resolve_cik(ticker: str,
  company_tickers: dict) -> str | None` (10-digit zero-padded CIK string),
  `captionlm.eval_dataset.find_press_release_filing(cik: str, call_date:
  str, submissions: dict, window_days: int = 5) -> dict | None` (keys
  `"cik"`, `"accession"`, `"primary_document"`),
  `captionlm.eval_dataset.build_filing_url(cik: str, accession: str,
  primary_document: str) -> str`, `captionlm.eval_dataset.fetch_json(url:
  str) -> dict` (thin `urllib` wrapper, not unit tested — see Step 5),
  `captionlm.eval_dataset.assemble_triple(ticker: str, call_date: str) ->
  dict | None`.

**Note on what's tested here:** `resolve_cik`, `find_press_release_filing`,
and `build_filing_url` are pure functions tested against fixture JSON, no
network. `fetch_json` and `assemble_triple` hit live SEC EDGAR endpoints —
not covered by an automated test (flaky/slow to run in CI against a live
government API); Step 5 is a manual verification run instead, which is
still concrete and checkable, just not part of `pytest`.

- [ ] **Step 1: Write the failing tests for the pure functions**

```python
# tests/test_eval_dataset.py
from captionlm.eval_dataset import build_filing_url, find_press_release_filing, resolve_cik

COMPANY_TICKERS_FIXTURE = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 1018724, "ticker": "AMZN", "title": "Amazon.com Inc."},
}

SUBMISSIONS_FIXTURE = {
    "filings": {
        "recent": {
            "form": ["10-K", "8-K", "4", "8-K"],
            "filingDate": ["2021-01-15", "2021-02-03", "2021-02-04", "2021-06-01"],
            "accessionNumber": ["0001-21-000001", "0001-21-000002", "0001-21-000003", "0001-21-000004"],
            "primaryDocument": ["10k.htm", "pressrelease.htm", "form4.xml", "other.htm"],
        }
    }
}


def test_resolve_cik_found():
    assert resolve_cik("aapl", COMPANY_TICKERS_FIXTURE) == "0000320193"


def test_resolve_cik_not_found():
    assert resolve_cik("nope", COMPANY_TICKERS_FIXTURE) is None


def test_find_press_release_filing_within_window():
    filing = find_press_release_filing("0000320193", "2021-02-01", SUBMISSIONS_FIXTURE, window_days=5)
    assert filing == {"cik": "0000320193", "accession": "0001-21-000002", "primary_document": "pressrelease.htm"}


def test_find_press_release_filing_outside_window():
    filing = find_press_release_filing("0000320193", "2021-05-01", SUBMISSIONS_FIXTURE, window_days=5)
    assert filing is None


def test_build_filing_url():
    url = build_filing_url("0000320193", "0001-21-000002", "pressrelease.htm")
    assert url == "https://www.sec.gov/Archives/edgar/data/320193/000121000002/pressrelease.htm"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_eval_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'captionlm.eval_dataset'`

- [ ] **Step 3: Write `captionlm/eval_dataset.py`**

```python
"""Assemble doc+audio+transcript eval triples: Earnings-21/22 audio and
transcripts (cloned locally via git-lfs, see the plan's Step 5) paired
with the matching SEC EDGAR earnings press release as the "doc" input.

SEC EDGAR's fair-use policy requires a descriptive User-Agent with a real
contact on every request — set captionlm.config.SEC_CONTACT before
running assemble_triple/fetch_json against the live API.
"""
import json
import urllib.request
from datetime import date

from captionlm.config import SEC_CONTACT

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"


def resolve_cik(ticker: str, company_tickers: dict) -> str | None:
    ticker = ticker.upper()
    for entry in company_tickers.values():
        if entry.get("ticker", "").upper() == ticker:
            return f"{entry['cik_str']:010d}"
    return None


def find_press_release_filing(
    cik: str, call_date: str, submissions: dict, window_days: int = 5
) -> dict | None:
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    target = date.fromisoformat(call_date)
    best = None
    best_delta = None
    for form, filing_date, accession, primary_doc in zip(forms, filing_dates, accessions, primary_docs):
        if form != "8-K":
            continue
        delta = abs((date.fromisoformat(filing_date) - target).days)
        if delta > window_days:
            continue
        if best_delta is None or delta < best_delta:
            best = {"cik": cik, "accession": accession, "primary_document": primary_doc}
            best_delta = delta
    return best


def build_filing_url(cik: str, accession: str, primary_document: str) -> str:
    accession_nodashes = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodashes}/{primary_document}"


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": SEC_CONTACT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def assemble_triple(ticker: str, call_date: str) -> dict | None:
    company_tickers = fetch_json(COMPANY_TICKERS_URL)
    cik = resolve_cik(ticker, company_tickers)
    if cik is None:
        return None

    submissions = fetch_json(SUBMISSIONS_URL_TEMPLATE.format(cik=cik))
    filing = find_press_release_filing(cik, call_date, submissions)
    if filing is None:
        return None

    filing["url"] = build_filing_url(filing["cik"], filing["accession"], filing["primary_document"])
    return filing
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_eval_dataset.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Manual verification against live SEC EDGAR**

First set a real contact in `captionlm/config.py`'s `SEC_CONTACT`, then:

```bash
venv/bin/python -c "
from captionlm.eval_dataset import assemble_triple
print(assemble_triple('AAPL', '2020-10-29'))
"
```

Expected: a dict with a `url` key pointing at a real SEC EDGAR filing
page for an Apple 8-K filed within 5 days of 2020-10-29 (an actual Apple
earnings date). If it prints `None`, try a different ticker/date — not
every company files a standalone 8-K press release (see the design doc's
"Ceiling, stated plainly" note); a few clean matches are enough.

- [ ] **Step 6: Clone Earnings-21 (audio + transcripts + entity tags)**

```bash
git clone --filter=blob:none --sparse https://github.com/revdotcom/speech-datasets.git vendor_data/speech-datasets
cd vendor_data/speech-datasets
git sparse-checkout set earnings21
git lfs pull
cd ../..
```

Expected: `vendor_data/speech-datasets/earnings21/` contains audio files
and transcripts for the 44 Earnings-21 calls. This step is environment
setup, not application code — nothing to unit test; confirm the directory
is non-empty and pick 5-10 calls across sectors to start with, per the
design doc.

- [ ] **Step 7: Commit**

```bash
echo "vendor_data/" >> .gitignore
git add captionlm/eval_dataset.py tests/test_eval_dataset.py .gitignore
git commit -m "$(cat <<'EOF'
Add SEC EDGAR filing lookup for automated eval dataset assembly

Pure ticker->CIK and CIK->matching-8-K logic is unit tested against
fixture JSON. Live EDGAR fetch (fetch_json/assemble_triple) is
verified manually per the plan, not in the automated test suite.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** every phase in the spec (env+vendor, term extraction,
  term graph, biased model + merge, CLI, eval dataset + harness) maps to a
  task above. The one deviation (`fscore.py` adapted, not vendored
  unmodified) is called out in Global Constraints and Task 1, with the
  reason.
- **Type consistency checked:** `SpotterConfig` is defined once (Task 2,
  `captionlm/config.py`) and imported everywhere it's used (Task 5).
  `blank_idx` is computed the same way (`len(model.vocabulary)`) in Tasks
  5, 6, and 7. `AlignedToken`/`WSHyp`/`ContextGraphCTC` names and fields
  are used consistently across Tasks 3-7, matching what Task 1 vendors and
  what `parakeet-mlx` actually exports (verified against the installed
  package's source, not guessed).
- **No placeholders:** every step above has real, complete code or an
  exact runnable command with a stated expected result.
