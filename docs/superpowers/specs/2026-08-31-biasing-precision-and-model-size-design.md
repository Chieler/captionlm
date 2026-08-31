# Biasing precision and model size — design

**Status:** ready for implementation (Parts 1-3); Part 4 needs a spike first
**Evidence:** `docs/results/2026-08-31-what-biasing-can-and-cannot-fix.md`

## Problem

On the read-aloud eval set, biasing at cb_weight 3.0 takes WER from
0.1160 to 0.0989 on `parakeet-tdt_ctc-110m`. That is a real gain, but the
error decomposition shows it is bought inefficiently and that most of the
remaining error is out of biasing's reach:

| model | baseline | biased | fixed | introduced | residual on terms | residual elsewhere |
|---|---|---|---|---|---|---|
| 110m | 144 | 122 | 59 | 37 | 35 | 87 |
| 1.1b | 125 | 96 | 45 | 16 | 33 | 63 |

Three separable problems:

1. **The acoustic model is the floor.** 63 of 96 residual errors at 1.1b
   are on words that are not domain terms. A measured swap to
   `parakeet-tdt_ctc-1.1b` moves unbiased WER 0.1160 -> 0.1009 and biased
   WER 0.0989 -> 0.0778, and is a config change, not a code change.
2. **`merge_spans` destroys words it did not replace.** 16 lossy
   replacements on this set, almost all `'the hard parts' -> 'hard
   parts'`: a keyword overlapping the preceding article by the 30%
   `intersection_threshold` consumes it.
3. **Extraction admits terms that cause substitutions** (`better`,
   `real`, `lets`, `traces`) and misses inflected forms of terms it does
   have (`subtrees` against a list holding `subtree`).

And one problem that none of the above touches: **homophones**. `Groq`
and `Grok` are both in the same term list and acoustically identical.
`write`/`right`, `weights`/`waits`. A context graph matches audio to
token sequences with no access to sentence-level context, so no term list
and no weight resolves these.

## Non-goals

- Training, fine-tuning, or adapting any model at use time. This is the
  project's central constraint and the reason CTC-WS was chosen: the term
  list compiles into a trie at call time. Every change below preserves it.
- Raising `TERM_EXTRACTION_TOP_N`. List size improves WER monotonically on
  the read-aloud set, but that measurement is confounded — the document
  *is* the transcript there, so fragments score as real terms. Not acted on.
- Beam or N-best decoding. `parakeet_mlx` implements `decode_beam`, but it
  returns a single hypothesis per batch item; exposing N-best means forking
  212 lines of decoder. Deferred, and Part 4 is designed not to need it.

## Part 1 — Model size as a supported choice

`MODEL_ID` becomes a default, not an assumption. `parakeet-tdt_ctc-1.1b`
is the same TDT-CTC architecture, so `BiasedParakeetTDTCTC`, the CTC head,
the sentencepiece vocabulary alignment and the spotter all work unchanged
— this was verified by running the full sweep against it.

Cost is real and belongs to the user, not to us: 4.0 GB on disk against
438 MB, and 28x realtime against 47x. Both remain far faster than
realtime, and neither trains anything. So the choice is exposed and
documented rather than switched silently.

The one hazard: `terms.py` downloads `tokenizer.model` for `model_id` and
`biased_model.py` downloads weights for `hf_id`. Nothing today asserts
these agree. A mismatched pair produces token ids that address the wrong
vocabulary and a context graph that silently never fires.

## Part 2 — Make `merge_spans` conservative

The rule today: find every word whose frame span overlaps the hypothesis
by `>= intersection_threshold`, replace the whole contiguous range with
one token carrying `hyp.word`.

The failure: overlap alone does not establish that a word is *part of*
the keyword. An article immediately before a keyword shares frames with
it and gets consumed.

The fix: a replacement may not destroy words. After the overlap range is
chosen, trim words from its edges until the range holds no more words
than `hyp.word` does. Trim the lower-overlap edge first — the true
keyword frames overlap most.

This is deliberately conservative. It cannot fix a keyword that should
have consumed a genuine multi-word span; it only prevents the range
growing beyond what the keyword can account for. The histogram says that
is the right trade: 152 clean 2-for-2 replacements against 14 that turn
three words into two.

## Part 3 — Extraction precision

Two independent changes to `captionlm/biasable.py` and `captionlm/terms.py`:

**Inflected variants.** `build_context_graph` already adds `{term,
term.lower()}`. Add regular English plural and third-person forms so
`subtree` in the list also matches spoken `subtrees`. Purely additive to
the trie, no extraction change, no new dependency (spaCy is already loaded
and gives lemmas).

**Prune terms that cause substitutions.** `better`, `real`, `lets` and
`traces` pass `is_rare` because they are uncommon in the pyate corpus, but
they are ordinary English words the model already transcribes. The rarity
threshold alone cannot separate them from `quorum`. Require a single-word
lowercase candidate to also not be a closed-class or high-frequency
lemma per spaCy's tagger.

## Part 4 — Document-conditioned rescoring (spike first, no plan yet)

The only lever that reaches both homophones and the 63 non-term residual
errors is a second pass with sentence-level context: hand the transcript
and the source document to an LLM and let it repair.

This is not planned here, deliberately. Writing task-level steps for it
now would require inventing an interface, a prompt, and an error budget
for a mechanism nobody has measured, which is how plans acquire
placeholders. It needs a spike answering one question first:

> Given the biased transcript and the source document, does an LLM
> actually fix `Grok` -> `Groq` without introducing new errors, and at
> what latency?

Run the spike against the two read-aloud clips, whose true answers are
known exactly. If the answer is yes, it earns its own spec and plan; the
architecture is a new module consuming `AlignedResult` and returning a
corrected one, so it does not disturb Parts 1-3.

## Verification

Every part is measured on the same harness that produced the evidence:
`scripts/sweep_cb_weight.py read_aloud --condition product=per-clip`.
A change that does not move WER or the introduced-error count on that set
is not accepted on argument alone.

Note the standing caveat: the read-aloud set is an upper bound (source
document identical to reference transcript, n=2). It is the right
instrument for "did this change help or hurt", and the wrong one for
quoting a user-facing accuracy number.
