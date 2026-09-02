# Can an LLM write the term list?

**No.** Measured three ways against the shipped pyate+spaCy extractor. It
never improves WER, it loses badly on read-aloud, and where it does gain
term recall it pays for it by doubling the injection rate.

The probe found something more useful than its own answer, recorded at the
bottom: **on the Earnings product clips, biasing with the shipped extractor
is worse than not biasing at all.**

Generator: `mlx-community/Qwen3-1.7B-4bit` (~1 GB, inside the 2 GB budget),
temperature 0, over each clip's own source document. Output filtered through
`captionlm.biasable.filter_biasable`, the same gate the shipped path uses, so
this compares extractors rather than comparing a filtered list to an
unfiltered one. Acoustic model `parakeet-tdt_ctc-110m` throughout.

## Method

Three arms over identical audio:

| arm | term list |
|---|---|
| `spacy` | what ships: pyate terminology + spaCy NER, `is_biasable` filtered |
| `llm` | Qwen3-1.7B-4bit over the same source document, same filter |
| `union` | both, deduplicated |

**Every arm is scored against one fixed term list — the union of all arms.**
`score()` measures recall over the list it is handed, so scoring each arm
against its own list would ask "of the terms you picked, how many did you
get". That flatters a short list and is not comparable across arms. The
unbiased arm is scored against the same fixed list, so all four rows compare.

## Results

**Earnings product clips** (`data/eval_data/clips`, 3 clips, 255 scoring terms).
These are SEC filings biasing the earnings calls they describe — the product
scenario, where the document is *not* the transcript.

| arm | WER | recall | precision | F | injection |
|---|---|---|---|---|---|
| unbiased | **0.1795** | 0.9136 | — | — | — |
| spacy | 0.1810 | 0.8959 | 0.8895 | 0.8927 | 0.065 |
| llm | 0.1806 | **0.9349** | 0.8634 | 0.8977 | 0.127 |
| union | 0.1820 | 0.9195 | 0.8266 | 0.8706 | 0.088 |

**Read-aloud** (`data/read_aloud`, 2 clips, 473 scoring terms). Clean
single-speaker audio where the document IS the transcript.

| arm | WER | recall | precision | F | injection |
|---|---|---|---|---|---|
| unbiased | 0.1160 | 0.7610 | — | — | — |
| spacy | 0.0969 | 0.9006 | 0.9496 | 0.9244 | n/a |
| llm | 0.1081 | 0.7954 | 0.9498 | 0.8658 | 0.000 |
| union | **0.0956** | **0.9044** | 0.9498 | **0.9265** | 0.000 |

## Reading them

**WER does not move.** Every Earnings arm sits within 0.0025 of every other,
and the read-aloud union beats spacy by 0.0013 over two clips and ~1500
reference words. Neither is a result at this sample size.

**The LLM alone is clearly worse on read-aloud** — recall 0.7954 against
0.9006, barely above the 0.7610 of no biasing at all.

That is not a list-size artifact, which was the obvious objection. The LLM
lists were 50 and 37 terms against pyate+spaCy's 206 and 242, so the arm was
re-run with the cap lifted and the chunk size halved. It produced 33 and 55.
**Short lists are intrinsic to the model at this size,** not imposed by the
probe. The first run's numbers are within noise of the second (llm recall
0.7950 against 0.7954), so nothing here turns on it.

**On Earnings the LLM does buy real recall** — 0.9349 against 0.8959, +0.039,
the only difference in either table large enough to trust besides the
read-aloud loss. It costs precision (0.8634 against 0.8895) and **doubles the
injection rate, 0.127 against 0.065**. That is the failure mode already on
record in `2026-08-31-what-biasing-can-and-cannot-fix.md`: 19 terms that never
produced a true positive accounted for 79 of 178 false accepts. An LLM asked
for plausible domain vocabulary is a machine for producing exactly those.

**Union never wins anything worth having.** Best F on read-aloud by 0.0021,
worst F on Earnings by 0.022.

**The two extractors barely overlap** — 1, 0 and 1 shared terms out of 50 on
the three Earnings clips. They are not variations on a theme. pyate returns
n-gram phrases; the LLM returns proper nouns. On read-aloud that showed as the
LLM finding `Claude`, `Gemini`, `Meta`, `Llama`, `Mistral`, `DeepSeek`,
`Cohere` and `Alibaba` where spaCy's NER had not, which is why the union arm
was worth running at all. It still did not pay.

## Two things the generator needed before it worked

Both are properties of small local models, and both cost more effort than the
result justified:

- **The documents had to be cleaned.** SEC filings arrive as HTML-to-text with
  runs of non-breaking spaces and table scaffolding. A 4000-word filing is
  7246 tokens, most of it whitespace, and the model echoed the document back
  instead of extracting from it.
- **The documents had to be chunked.** At ~5000 tokens of context Qwen3-1.7B
  stopped following the instruction and emitted sentence fragments
  (`% for brakes compared to the prior year period.`). At ~600 words it
  extracted. It also echoed its own instructions back as findings, so
  `technical jargon` and `acronyms` had to be filtered out by name.

## The finding that matters, which is not about the LLM

**On the Earnings product clips, biasing makes things worse than not biasing.**
Term recall 0.8959 against 0.9136 unbiased. WER 0.1810 against 0.1795. The
shipped extractor's list is actively harmful there, and its top-50 explains
why:

```
+
long
shares outstanding
lesser extent
line
line results in the second quarter
line with key industry indicators
```

pyate's n-grams over an SEC filing, with the 50-term cap spent on
function-word phrases. This is a live problem in the product path — the
`caption_dropoff` and dashboard flows use `extract_bias_terms` on exactly this
kind of document — and it is independent of anything an LLM does. It has not
been fixed and is not tracked anywhere else. Worth its own investigation.

Note this does not contradict the read-aloud results, where the same extractor
takes WER from 0.1160 to 0.0969. There the document is the transcript, and
pyate's phrases are drawn from the words actually spoken.

## Caveats

Three Earnings clips and two read-aloud clips, one acoustic model (110m), one
generator at temperature 0, one prompt. Differences under ~0.01 WER are noise
at this size. The two differences large enough to act on are the Earnings
recall gain (+0.039, bought with doubled injections) and the read-aloud recall
loss (−0.105).

The probe scripts were throwaway and are not in the repository.
