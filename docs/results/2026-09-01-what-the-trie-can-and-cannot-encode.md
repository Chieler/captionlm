# What the context trie can and cannot encode

Two questions about the CTC-WS graph, both answered by measurement on
`parakeet-tdt_ctc-1.1b` over `read_aloud/ai_agents.wav`.

## 1. Is the blank token in the trie? Yes, and the topology is right

CTC alignments emit blanks between and inside tokens, so a trie with only
token arcs would lose every token pass at the first blank frame and spot
nothing at all. Audited over a graph built from the real 1024-piece
sentencepiece vocabulary:

- 46 reachable nodes: 27 token nodes, 18 blank nodes, 1 root.
- Every token node with a forward transition also has a blank arc.
- Every blank node self-loops on blank, so any number of blank frames is
  absorbed.
- The blank detour rejoins the *same* next state as the direct arc — it is
  a detour, not a parallel branch that would strand the word-final state.
- Every token node self-loops on its own token, so a token held across
  frames survives.
- The root has **no** blank arc, which is correct: `run_word_spotter`
  seeds a fresh empty token at the root on every frame, so a blank
  self-loop there would only duplicate that.

`blank_id` is `len(model.vocabulary)` = 1024, and `build_context_graph`
already refuses to build if that disagrees with the tokenizer's piece
count. Locked down by `test_the_trie_carries_the_blank_token_per_ctc_topology`
and `test_the_blank_detour_rejoins_the_same_next_state`, because this is
an invariant that fails silently — a broken graph does not error, it just
stops firing.

## 2. Would embedding meaning in the trie pick the right word? No

The test case is the one the eval keeps failing: `Groq` and `Grok` are
both in `ai_agents.terms.txt`, both spoken in the same sentence, and
acoustically identical. They tokenize to `[▁G, ro, q]` and `[▁G, ro, k]`
— one differing token, the last, so they share a trie prefix and diverge
on a single arc.

Spotter output, both occurrences, `cb_weight` 3.0:

| region | truth | context | `Grok` | `Groq` | margin |
|---|---|---|---|---|---|
| 0 | **Groq** | `__ built custom silicon` | 6.946 | 4.048 | −2.898 |
| 1 | **Grok** | `confusion with __` | 7.624 | 3.540 | −4.084 |

Both are spotted on the identical frames (1202–1205, 1347–1351). `Grok`
wins both, so region 0 is wrong. The per-frame acoustic evidence at the
deciding frame prefers `k` over `q` by +2.898 and +4.084 — that is, it
prefers `k` **in the place where the speaker said Groq**. The model's own
argmax at both frames is `ck`: it wants to write "Grock" either way.

Four reasons no amount of meaning in the trie fixes this.

**There is nowhere to put a vector.** The only per-token quantity in the
spotter is `token.score + logprobs[frame][token] + cb_weight`: a scalar
added to a per-frame acoustic log-probability. There is no hidden state
and no context vector to compute a similarity against. Any "meaning"
attached to a trie node collapses to one number per term, because one
number per term is all the search can consume.

**CTC has no output context to condition on — that is the point of it.**
CTC assumes conditional independence across frames: each frame's
distribution depends only on the encoder output at that frame, never on
previously emitted tokens. It carries no internal language model at all.
The transducer is the head with output-side context, through its
prediction network. This is exactly why CTC-WS is safe to bias — there is
no autoregressive state to corrupt, which is why `generate()` can spot
over the CTC head and splice into an untouched TDT decode — and exactly
why the spotter cannot disambiguate: there is no context inside it.

**The information needed is positional; a trie weight is global.**
Occurrence 1 is Groq, occurrence 2 is Grok, at near-identical acoustic
scores. One constant per term cannot say "the first one, not the second".
A feasible window does exist here — a boost `D` on `Groq` fixes region 0
if `D > 2.898` and leaves region 1 correct if `D < 4.084` — but that
1.19-wide window is an accident of the acoustics, found by looking at the
answer, and carries no positional information whatsoever. A clip with two
Groqs and one Grok breaks it.

**Meaning cannot rank the pair anyway.** The only semantic source is the
document, and the document contains both terms — the sentence being
transcribed literally says "the name is a constant source of confusion
with Grok". Document-level salience is near-identical for exactly the
pair it would need to separate. That is not bad luck: two terms are
confusable *and* both listed precisely when the document discusses both.

What does separate them is the sentence: "___ built custom silicon for
low latency decoding" against "confusion with ___". That context exists
only once the transcript does, which is document-conditioned rescoring,
item 2 of the WER budget. This measurement is the strongest evidence yet
for it — the failure is fully diagnosed and provably out of reach of
every acoustic and trie-side lever.

### Where per-term weighting is still worth trying

Not for confusable pairs, for the "garbled listed term" bucket:
`linearizability` → `delarizability`, `replica` → `bobpeka`. Those are
not two candidates competing — they are one listed term losing to no
listed term at all, which a bigger push can genuinely fix. That remains
item 3 of the budget, and it is a length-and-rarity decision, not a
semantic one.
