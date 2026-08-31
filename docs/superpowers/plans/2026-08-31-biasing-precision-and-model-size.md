# Biasing Precision and Model Size Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop contextual biasing from destroying words it did not replace, make the larger acoustic model a supported choice, and tighten term extraction — measured on the read-aloud eval set.

**Architecture:** No new subsystems. `merge_spans` gains a trimming rule so a spotted keyword cannot consume more words than it accounts for. `MODEL_ID` becomes an explicit choice with a guard against tokenizer/weight mismatch. `terms.py` adds inflected surface forms to the trie and `biasable.py` stops admitting ordinary English words that the rarity gate alone lets through. The CTC-WS design is unchanged: the term list still compiles into a trie at call time and nothing is trained.

**Tech Stack:** Python 3.12, MLX, `parakeet-mlx` (`ParakeetTDTCTC`), spaCy `en_core_web_sm`, pyate, jiwer 4.0.0, pytest. Everything already installed; this plan adds no dependency.

**Spec:** `docs/superpowers/specs/2026-08-31-biasing-precision-and-model-size-design.md`
**Evidence:** `docs/results/2026-08-31-what-biasing-can-and-cannot-fix.md`

## Global Constraints

- **No training, fine-tuning, or model adaptation at use time, ever.** The user must never wait for a model to train. This is why CTC-WS was chosen: the term list compiles into a `ContextGraphCTC` trie at call time in microseconds. Any change that introduces a training step is rejected regardless of its accuracy.
- **No new dependencies.** spaCy, pyate, jiwer, sentencepiece, parakeet-mlx and mlx are already installed; use them.
- **`TERM_EXTRACTION_TOP_N` stays at 50.** List size improves WER on the read-aloud set, but that measurement is confounded (the document *is* the transcript there, so sentence fragments score as real terms). Do not raise it in this plan.
- **Every accuracy claim is measured, not argued.** The harness is `PYTHONPATH=. venv/bin/python scripts/sweep_cb_weight.py read_aloud --condition product=per-clip --score-terms eval_data/read_aloud_terms/score_all.txt`. A task that claims an improvement without a number from that harness is not done.
- **Run tests with `PYTHONPATH=. venv/bin/python -m pytest`.** Bare `pytest` fails with `ModuleNotFoundError: No module named 'captionlm'`.
- **Baselines to beat**, `parakeet-tdt_ctc-110m`, cb_weight 3.0, per-clip terms: WER 0.0989, term F 0.9301, 37 introduced errors. On `parakeet-tdt_ctc-1.1b`: WER 0.0778, term F 0.9464, 16 introduced errors.

---

### Task 1: Stop `merge_spans` consuming words the keyword does not account for

`merge_spans` selects every word overlapping a spotted hypothesis by at least `intersection_threshold` (30%) and replaces the whole contiguous range with one token holding `hyp.word`. An article immediately before a keyword shares frames with it, clears 30%, and is destroyed. Measured on both read-aloud clips at cb_weight 3.0: 16 lossy replacements, 16 words destroyed, of the shape `'the hard parts' -> 'hard parts'`, `'a small model' -> 'small model'`, `'a fencing token,' -> 'fencing token'`.

The rule to add: a replacement may not shrink the word count. Trim words off the edges of the chosen range, lowest-overlap edge first, until the range holds no more words than `hyp.word` does.

**Files:**
- Modify: `captionlm/merge.py:41-96` (the `merge_spans` body; `_overlap_pct` at :35 is reused unchanged)
- Test: `tests/test_merge.py`

**Interfaces:**
- Consumes: `merge_spans(tokens, ws_hyps, time_ratio, intersection_threshold=30.0) -> list[AlignedToken]`, `_overlap_pct(word_start, word_end, hyp_start, hyp_end) -> float`, `_group_words(tokens) -> list[list[AlignedToken]]`, all already in `captionlm/merge.py`. `WSHyp(word: str, score: float, start_frame: int, end_frame: int)` from `captionlm.vendor.ctc_word_spotter`.
- Produces: `merge_spans` keeps its exact signature and return type. No caller changes.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_merge.py` (the file already defines `_tok` and `TIME_RATIO = 0.08`; reuse them, do not redefine):

```python
def test_leading_article_is_not_consumed_by_a_shorter_keyword():
    # Measured failure: 'the hard parts' -> 'hard parts'. The article's
    # frames overlap the keyword enough to clear intersection_threshold,
    # and the whole range was replaced by the keyword alone.
    tokens = [
        _tok(" the", 0.16, 0.16),
        _tok(" hard", 0.32, 0.16),
        _tok(" parts", 0.48, 0.16),
    ]
    # hyp covers frames [3, 7] = 0.24s..0.64s, which clips half of " the"
    hyps = [WSHyp(word="hard parts", score=9.0, start_frame=3, end_frame=7)]

    merged = merge_spans(tokens, hyps, TIME_RATIO)

    assert "".join(t.text for t in merged).split() == ["the", "hard", "parts"]


def test_trims_the_lower_overlap_edge_first():
    # Both edges clear the threshold, so the range holds three words for a
    # two-word keyword. " a" overlaps the hypothesis 50%, " token" 100%,
    # so " a" is the one that is not part of the keyword.
    tokens = [
        _tok(" a", 0.00, 0.16),
        _tok(" fencing", 0.16, 0.24),
        _tok(" token", 0.40, 0.24),
        _tok(" now", 0.64, 0.16),
    ]
    # frames [1, 7] = 0.08s..0.64s, which clips half of " a"
    hyps = [WSHyp(word="fencing token", score=9.0, start_frame=1, end_frame=7)]

    merged = merge_spans(tokens, hyps, TIME_RATIO)

    assert "".join(t.text for t in merged).split() == [
        "a", "fencing", "token", "now",
    ]


def test_exact_width_multiword_replacement_still_works():
    # The conservative rule must not break the case it was built for:
    # two spoken words replaced by a two-word keyword.
    tokens = [
        _tok(" hello", 0.0, 0.16),
        _tok(" san", 0.16, 0.16),
        _tok(" francisco", 0.32, 0.32),
        _tok(" today", 0.64, 0.16),
    ]
    hyps = [WSHyp(word="San Francisco", score=8.0, start_frame=2, end_frame=7)]

    merged = merge_spans(tokens, hyps, TIME_RATIO)

    assert "".join(t.text for t in merged).split() == [
        "hello", "San", "Francisco", "today",
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/test_merge.py -v`
Expected: `test_leading_article_is_not_consumed_by_a_shorter_keyword` FAILS, asserting `['hard', 'parts'] == ['the', 'hard', 'parts']`. `test_exact_width_multiword_replacement_still_works` PASSES already — it is the regression guard, and it must stay passing.

- [ ] **Step 3: Add the trimming rule**

In `captionlm/merge.py`, add this helper immediately after `_overlap_pct`:

```python
def _trim_to_keyword_width(
    overlap_idxs: list[int],
    words: list[list[AlignedToken]],
    hyp_start: float,
    hyp_end: float,
    keyword_words: int,
) -> list[int]:
    """Shrink an overlap range until it holds no more words than the
    keyword does, dropping the lower-overlap edge each time.

    Frame overlap alone does not prove a word is part of the keyword. An
    article immediately before one shares frames with it and clears the
    30% threshold, and replacing the whole range destroys it: measured on
    the read-aloud set, 16 replacements lost a word each, almost all of
    the shape 'the hard parts' -> 'hard parts'. The keyword's own word
    count is the honest bound on how much it can account for.
    """
    idxs = list(overlap_idxs)
    while len(idxs) > keyword_words:
        first, last = words[idxs[0]], words[idxs[-1]]
        head = _overlap_pct(first[0].start, first[-1].end, hyp_start, hyp_end)
        tail = _overlap_pct(last[0].start, last[-1].end, hyp_start, hyp_end)
        idxs.pop(0 if head <= tail else -1)
    return idxs
```

Then in `merge_spans`, replace this block:

```python
        if not overlap_idxs:
            continue

        first_idx, last_idx = overlap_idxs[0], overlap_idxs[-1]
```

with:

```python
        overlap_idxs = _trim_to_keyword_width(
            overlap_idxs, words, hyp_start, hyp_end, len(hyp.word.split())
        )
        if not overlap_idxs:
            continue

        first_idx, last_idx = overlap_idxs[0], overlap_idxs[-1]
```

Note the ordering: `overlap_idxs` is contiguous by construction (it is built from a single scan over `words`), so popping from either end keeps it contiguous.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/test_merge.py -v`
Expected: all PASS, including the pre-existing `test_single_word_replacement` and `test_multiword_span_replacement`.

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/ -q`
Expected: all PASS (68 before this task).

- [ ] **Step 6: Measure on the real harness**

Run:

```bash
PYTHONPATH=. venv/bin/python scripts/sweep_cb_weight.py read_aloud \
  --condition product=per-clip \
  --score-terms eval_data/read_aloud_terms/score_all.txt \
  --weights 3.0 --out-dir eval_results/task1
```

Expected: WER at or below 0.0989 (the 110m baseline with per-clip terms). Record the exact number in the commit message. If WER got **worse**, stop and report — the trimming rule is dropping a real multi-word replacement, and the fix is to trim only when the range is strictly longer than the keyword, which the code above already does; a regression means the assumption about contiguity or word counting is wrong.

- [ ] **Step 7: Commit**

```bash
git add captionlm/merge.py tests/test_merge.py
git commit -m "Stop a spotted keyword consuming words it does not replace"
```

---

### Task 2: Guard against a tokenizer/weights mismatch, and support the 1.1b model

`load_biased_model(hf_id)` downloads weights for one model id; `load_tokenizer(model_id)` downloads `tokenizer.model` for another. Nothing asserts they agree. A mismatched pair yields token ids addressing the wrong vocabulary and a context graph that silently never fires — no error, just biasing that does nothing.

This matters now because `parakeet-tdt_ctc-1.1b` becomes a supported choice: measured on the read-aloud set it gives unbiased WER 0.1009 against 110m's 0.1160, and biased 0.0778 against 0.0989. It is the same TDT-CTC architecture, so no code path changes — but it makes passing two different ids to two different functions a live hazard rather than a theoretical one.

**Files:**
- Modify: `captionlm/config.py:4` (add the model constants and a docstring)
- Modify: `captionlm/terms.py:27-38` (`build_context_graph` gains the vocabulary check)
- Test: `tests/test_terms.py`

**Interfaces:**
- Consumes: `load_tokenizer(model_id) -> spm.SentencePieceProcessor`, `build_context_graph(terms, tokenizer, blank_idx) -> ContextGraphCTC`, both in `captionlm/terms.py`. `model.vocabulary` is a `list[str]` on `BiasedParakeetTDTCTC`; `blank_idx` is always `len(model.vocabulary)`.
- Produces: `build_context_graph` keeps its signature and return type; it raises `ValueError` on mismatch instead of returning a silently dead graph. `captionlm.config.MODEL_ID_1_1B` is a new module constant available to `cli.py`, `eval.py` and `scripts/sweep_cb_weight.py` via their existing `--model` flags.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_terms.py`:

`tests/test_terms.py:1-2` already imports `MODEL_ID`, `build_context_graph`
and `load_tokenizer`. Add only `import pytest` at the top, then the tests:

```python
def test_build_context_graph_rejects_a_blank_idx_from_a_different_model():
    # A tokenizer from one model and a blank_idx from another produce token
    # ids that address the wrong vocabulary. The graph would build without
    # complaint and then never fire, which is the worst possible failure:
    # biasing that silently does nothing.
    tokenizer = load_tokenizer(MODEL_ID)
    wrong_blank_idx = tokenizer.get_piece_size() + 500

    with pytest.raises(ValueError, match="blank_idx"):
        build_context_graph(["Kubernetes"], tokenizer, wrong_blank_idx)


def test_build_context_graph_accepts_the_matching_blank_idx():
    tokenizer = load_tokenizer(MODEL_ID)
    graph = build_context_graph(["Kubernetes"], tokenizer, tokenizer.get_piece_size())
    assert graph is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/test_terms.py -v`
Expected: `test_build_context_graph_rejects_a_blank_idx_from_a_different_model` FAILS with `DID NOT RAISE <class 'ValueError'>`.

- [ ] **Step 3: Add the check and the model constants**

In `captionlm/terms.py`, at the top of `build_context_graph`, before the loop:

```python
    if blank_idx != tokenizer.get_piece_size():
        raise ValueError(
            f"blank_idx {blank_idx} does not match this tokenizer's vocabulary "
            f"size {tokenizer.get_piece_size()}. The tokenizer and the model "
            f"weights come from different model ids; the token ids would "
            f"address the wrong vocabulary and the graph would never fire."
        )
```

In `captionlm/config.py`, replace line 4 with:

```python
# Both are hybrid TDT-CTC, which is what the CTC-WS spotter requires: a
# TDT-only or CTC-only checkpoint has no compatible pair of heads. Measured
# on the read-aloud eval set at cb_weight 3.0 -- 110m: 0.1160 unbiased,
# 0.0989 biased, 438 MB, ~47x realtime. 1.1b: 0.1009 unbiased, 0.0778
# biased, 4.0 GB, ~28x realtime. Neither trains anything at use time.
MODEL_ID_110M = "mlx-community/parakeet-tdt_ctc-110m"
MODEL_ID_1_1B = "mlx-community/parakeet-tdt_ctc-1.1b"

# The small model stays the default: 4.0 GB is the user's disk and the
# user's download, and that is their choice to make with --model, not ours
# to make silently on their behalf.
MODEL_ID = MODEL_ID_110M
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/test_terms.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/ -q`
Expected: all PASS. `tests/test_biased_model.py` builds a real graph from a real model; if it now raises `ValueError`, the check is wrong, not the test — `blank_idx` there is `len(model.vocabulary)` and must equal `tokenizer.get_piece_size()`.

- [ ] **Step 6: Confirm the 1.1b path works end to end**

Run:

```bash
PYTHONPATH=. venv/bin/python -m captionlm.cli read_aloud/ai_agents.wav \
  --terms read_aloud/ai_agents.terms.txt \
  --model mlx-community/parakeet-tdt_ctc-1.1b \
  --out /tmp/ai_1b.srt
```

Expected: writes `/tmp/ai_1b.srt`, no `ValueError`. First run downloads 4.0 GB. Spot-check that the file contains `Groq` or `Anthropic` — if the SRT is empty or the terms are absent, the graph is not firing.

- [ ] **Step 7: Commit**

```bash
git add captionlm/config.py captionlm/terms.py tests/test_terms.py
git commit -m "Name both TDT-CTC checkpoints and refuse a mismatched tokenizer"
```

---

### Task 3: Match inflected forms of listed terms

The term list holds `subtree`; the speaker says `subtrees`; the spotter does not fire. Measured residuals of exactly this shape: `subtrees` -> `subtree`, `objects` -> `object`, `shorter` -> `short`, `monotonic reads` -> `monotonistic read`.

`build_context_graph` already adds two surface variants per term (`{term, term.lower()}`). Add regular English inflections to the same set. This is additive to the trie only — no extraction change, no new dependency, and the trie build stays microseconds, so the no-training constraint is untouched.

Scope it deliberately: regular `-s`/`-es` on the final word only. Irregular plurals and comparatives (`shorter`) are not handled; a real morphological generator is a bigger change than the residual justifies.

**Files:**
- Modify: `captionlm/terms.py:27-38` (`build_context_graph`, after Task 2's check)
- Test: `tests/test_terms.py`

**Interfaces:**
- Consumes: `build_context_graph(terms, tokenizer, blank_idx)` including the `ValueError` guard added in Task 2. Do not remove that guard.
- Produces: a new module-level `_surface_variants(term: str) -> set[str]` in `captionlm/terms.py`, returning every surface form to enter into the trie for one term.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_terms.py`:

```python
from captionlm.terms import _surface_variants


def test_surface_variants_adds_a_regular_plural_of_the_last_word():
    assert "subtrees" in _surface_variants("subtree")
    assert "vector clocks" in _surface_variants("vector clock")


def test_surface_variants_uses_es_after_a_sibilant():
    assert "witnesses" in _surface_variants("witness")
    assert "boxes" in _surface_variants("box")


def test_surface_variants_of_an_already_plural_term_stays_inert():
    # "subtrees" gains "subtreeses", which no speaker will ever say, so the
    # extra trie entry cannot fire. Cheaper than detecting real plurals, and
    # the alternative -- skipping words ending in s -- would also lose
    # "witness" -> "witnesses".
    variants = _surface_variants("subtrees")
    assert "subtrees" in variants


def test_surface_variants_keeps_the_original_and_its_lowercase():
    variants = _surface_variants("MONRO FORWARD")
    assert "MONRO FORWARD" in variants
    assert "monro forward" in variants
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/test_terms.py -v`
Expected: FAIL with `ImportError: cannot import name '_surface_variants'`.

- [ ] **Step 3: Implement `_surface_variants` and use it**

In `captionlm/terms.py`, add above `build_context_graph`:

```python
_SIBILANT_ENDINGS = ("s", "x", "z", "ch", "sh")


def _surface_variants(term: str) -> set[str]:
    """Every spoken surface form of one term to enter into the trie.

    The list holds "subtree" and the speaker says "subtrees", so the
    spotter never fires. Regular -s/-es on the final word covers the
    measured cases; irregular plurals and comparatives ("shorter") are
    left alone, because a real morphological generator is a larger change
    than the remaining residual justifies.

    An already-plural term gains a nonsense variant -- "subtrees" yields
    "subtreeses". That is deliberate: no speaker says it, so the extra
    trie entry can never fire, and detecting real plurals costs more than
    an inert entry does. Do not "fix" it by skipping words ending in s;
    that would drop "witness" -> "witnesses" too.
    """
    variants = {term, term.lower()}
    for base in (term, term.lower()):
        head, _, last = base.rpartition(" ")
        if not last:
            continue
        plural = last + ("es" if last.endswith(_SIBILANT_ENDINGS) else "s")
        variants.add(f"{head} {plural}" if head else plural)
    return variants
```

Then in `build_context_graph`, replace:

```python
        variants = {term, term.lower()}
```

with:

```python
        variants = _surface_variants(term)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/test_terms.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 6: Measure on the real harness**

Run:

```bash
PYTHONPATH=. venv/bin/python scripts/sweep_cb_weight.py read_aloud \
  --condition product=per-clip \
  --score-terms eval_data/read_aloud_terms/score_all.txt \
  --weights 3.0 --out-dir eval_results/task3
```

Expected: WER at or below the number Task 1 recorded. More trie entries means more chances to false-accept, so a small regression is possible; if WER rises by more than 0.002, report it with the number rather than proceeding — the plural variants are then costing more than they recover, and the honest outcome is to revert this task, not to defend it.

- [ ] **Step 7: Commit**

```bash
git add captionlm/terms.py tests/test_terms.py
git commit -m "Match regular plurals of listed terms"
```

---

### Task 4: Stop extraction admitting ordinary English words

`better`, `real`, `lets` and `traces` are in the extracted term lists and cause substitutions of their own. They pass `is_rare` because they are uncommon in pyate's 1.56M-token corpus, but they are ordinary English the model already transcribes correctly. Rarity alone cannot separate them from `quorum`.

What separates them: they are not nouns. `quorum`, `tombstone`, `linearizability` and `sharding` are nouns; `better` is an adjective, `lets` a verb, `real` an adjective. `traces` is a genuine noun and will survive — that one needs the frequency signal, not the tagger, and is left alone.

**Files:**
- Modify: `captionlm/biasable.py` (the rare-lowercase harvest inside `_candidates`)
- Test: `tests/test_biasable.py`

**Interfaces:**
- Consumes: `_candidates(doc)` is a generator over one parsed spaCy `Doc`, yielding candidate surface strings; `is_rare(word) -> bool`; `is_biasable(term, nlp=None) -> bool`. All in `captionlm/biasable.py`.
- Produces: no signature changes. `extract_entities(text, top_n=None) -> list[str]` returns strictly fewer terms.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_biasable.py`:

```python
def test_rare_but_ordinary_non_nouns_are_not_harvested():
    # All four were extracted from the read-aloud scripts and caused
    # substitutions. They clear the rarity bar because pyate's corpus is
    # small, but they are ordinary English the model already gets right.
    text = (
        "Guardrails catch some of this. Better tool design catches more. "
        "Snapshot isolation lets readers see a consistent view of real data."
    )
    found = {t.lower() for t in extract_entities(text)}
    assert "better" not in found
    assert "lets" not in found
    assert "real" not in found


def test_rare_nouns_are_still_harvested():
    text = (
        "The protocol requires a quorum of members. A tombstone survives "
        "until every replica has seen it, and sharding spreads the load."
    )
    found = {t.lower() for t in extract_entities(text)}
    assert {"quorum", "tombstone", "sharding"} <= found
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/test_biasable.py -v`
Expected: `test_rare_but_ordinary_non_nouns_are_not_harvested` FAILS — at least one of `better`, `lets`, `real` is present. `test_rare_nouns_are_still_harvested` PASSES already and is the regression guard.

- [ ] **Step 3: Require the rare lowercase harvest to be nominal**

In `captionlm/biasable.py`, in `_candidates`, replace:

```python
    for token in tokens:
        if token.is_alpha and token.text.islower() and is_rare(token.text):
            yield token.text
```

with:

```python
    for token in tokens:
        # Nominal only. Rarity alone cannot separate "quorum" from "better":
        # both are uncommon in pyate's 1.56M-token corpus, but only one is a
        # term. The adjectives and verbs that slipped through -- better,
        # lets, real -- caused substitutions of their own on the read-aloud
        # set. A genuine rare noun that the model already gets right, such
        # as "traces", still survives this and needs the frequency signal
        # rather than the tagger.
        if (
            token.is_alpha
            and token.text.islower()
            and token.pos_ in ("NOUN", "PROPN")
            and is_rare(token.text)
        ):
            yield token.text
```

Update harvest 3's line in the `_candidates` docstring from:

```
    3. Rare lowercase words: quorum, tombstone, sharding.
```

to:

```
    3. Rare lowercase nouns: quorum, tombstone, sharding. Nominal because
       rarity alone cannot separate those from "better" and "lets", which
       are equally rare in the reference corpus and caused substitutions.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/test_biasable.py -v`
Expected: all PASS. If `test_extract_entities_finds_multiword_lowercase_jargon` now fails, the bigram harvest was depending on the single-token harvest for its anchors — it is not, it calls `is_rare` directly, so investigate rather than relaxing the new filter.

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 6: Regenerate the term lists and measure**

```bash
for d in ai_agents distributed_systems; do
  PYTHONPATH=. venv/bin/python scripts/extract_terms.py read_aloud/$d.txt \
    --mode combined --top-n 10000 --out eval_data/read_aloud_terms/$d.call.txt
  cp eval_data/read_aloud_terms/$d.call.txt read_aloud/$d.terms.txt
done
cat eval_data/read_aloud_terms/*.call.txt | sort -uf > eval_data/read_aloud_terms/score_all.txt

PYTHONPATH=. venv/bin/python scripts/sweep_cb_weight.py read_aloud \
  --condition product=per-clip \
  --score-terms eval_data/read_aloud_terms/score_all.txt \
  --weights 3.0 --out-dir eval_results/task4
```

Expected: list sizes drop from 206 and 242. WER at or below Task 3's number. Note that `--score-terms` changes here, so the F-score is measured against a different vocabulary than in Tasks 1 and 3 and is **not** comparable across those tasks; WER is.

- [ ] **Step 7: Commit**

```bash
git add captionlm/biasable.py tests/test_biasable.py
git commit -m "Harvest only nominal rare words, not every rare word"
```

---

### Task 5: Record the combined result

Four changes have landed, each measured alone. Record what they are worth together, at both model sizes, so the next plan argues from a current number rather than this plan's starting one.

**Files:**
- Modify: `docs/results/2026-08-31-what-biasing-can-and-cannot-fix.md` (append a section; do not edit the existing measurements — they are the before)

**Interfaces:**
- Consumes: `scripts/sweep_cb_weight.py`'s `--condition NAME=per-clip` form and its `--model` flag.
- Produces: nothing code depends on.

- [ ] **Step 1: Run both models on the final code**

```bash
PYTHONPATH=. venv/bin/python scripts/sweep_cb_weight.py read_aloud \
  --condition product=per-clip \
  --score-terms eval_data/read_aloud_terms/score_all.txt \
  --weights 2.0,3.0,4.0 --out-dir eval_results/final_110m

PYTHONPATH=. venv/bin/python scripts/sweep_cb_weight.py read_aloud \
  --condition product=per-clip \
  --score-terms eval_data/read_aloud_terms/score_all.txt \
  --weights 2.0,3.0,4.0 --model mlx-community/parakeet-tdt_ctc-1.1b \
  --out-dir eval_results/final_1b
```

- [ ] **Step 2: Recount introduced errors**

```bash
PYTHONPATH=. venv/bin/python - <<'PY'
import json, jiwer, os
from captionlm.eval import _NORMALIZE
def spans(ref, hyp):
    o = jiwer.process_words([ref], [hyp],
                            reference_transform=_NORMALIZE,
                            hypothesis_transform=_NORMALIZE)
    r = o.references[0]
    return {(c.type, " ".join(r[c.ref_start_idx:c.ref_end_idx]), c.ref_start_idx)
            for c in o.alignments[0] if c.type != "equal"}
for tag in ("final_110m", "final_1b"):
    d = json.load(open(f"eval_results/{tag}/sweep-product-3.0.json"))
    fixed = introduced = 0
    for p in d["per_clip"]:
        b = spans(p["reference"], p["baseline_pred"])
        x = spans(p["reference"], p["biased_pred"])
        fixed += len(b - x); introduced += len(x - b)
    print(f"{tag}: WER {d['baseline']['wer']:.4f} -> {d['biased']['wer']:.4f}, "
          f"fixed {fixed}, introduced {introduced}")
PY
```

Expected: `introduced` below 37 for 110m and below 16 for 1.1b. Those are the numbers Task 1 exists to move; if they did not move, say so plainly in the appended section rather than reporting only the WER.

- [ ] **Step 3: Append the results section**

Append to `docs/results/2026-08-31-what-biasing-can-and-cannot-fix.md` a section titled `## After the 2026-08-31 precision plan`, carrying: the two WER numbers from Step 1, the fixed/introduced counts from Step 2, the term-list sizes after Task 4, and one sentence per task saying what it was worth. Where a task was worth nothing or was negative, write that — a plan's own results doc is the one place a null result must not be quietly dropped.

- [ ] **Step 4: Commit**

```bash
git add docs/results/2026-08-31-what-biasing-can-and-cannot-fix.md
git commit -m "Record what the precision plan was worth, at both model sizes"
```

---

## Not in this plan

**Document-conditioned rescoring.** The only lever that reaches homophones (`Groq` and `Grok` are both in the term list and acoustically identical) and the 63 non-term residual errors is a second pass with sentence-level context. It is deliberately unplanned: writing task-level steps now would mean inventing an interface, a prompt and an error budget for a mechanism nobody has measured, which is how plans acquire placeholders.

It needs a spike first, against the two read-aloud clips whose true answers are known exactly, answering: given the biased transcript and the source document, does an LLM fix `Grok` -> `Groq` without introducing new errors, and at what latency? If yes, it earns its own spec and plan. The architecture would be a new module consuming an `AlignedResult` and returning a corrected one, so it does not disturb anything above.

**N-best decoding.** `parakeet_mlx` implements `decode_beam`, but it returns one hypothesis per batch item. Exposing N-best means forking 212 lines of decoder, and the spike above is designed not to need it.

**Raising `TERM_EXTRACTION_TOP_N`.** Confounded by the read-aloud set's upper-bound property. See the Global Constraints.
