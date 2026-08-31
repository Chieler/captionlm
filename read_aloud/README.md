# Read-aloud eval set

Two technical scripts, recorded by a human reader. Each `.txt` is
simultaneously the **source document** the term extractor reads and the
**reference transcript** the WER is scored against.

That identity is the point. On Earnings-21 we measured a hard ceiling:
across three calls, 47 ASR errors were fixable in principle and only 5
appeared anywhere in the company's own press release. Eleven percent. No
extractor, however good, can beat that — the words simply are not in the
document. Here they are, by construction, so a failure is attributable to
the extractor or the CTC-WS spotter rather than to the data source.

The flip side: this is an **upper bound, not a product estimate**. A real
user's document is related to their audio, not identical to it. Treat a
result here as "does the mechanism work at all", never as "this is what
users will get".

## The two scripts

| File | Profile | Planted confusables |
|---|---|---|
| `ai_agents.txt` | Organisation and product names — the NER stream's territory | Groq vs Grok (both present), LoRA vs Laura, Cohere, Mistral, Ollama vs "a llama", DeepSeek, Anthropic, Nvidia |
| `distributed_systems.txt` | Technical jargon and researcher surnames — the terminology stream's territory | Raft vs raft, Paxos, Lamport, Merkle, Byzantine, quorum, linearizability, tombstone, write skew |

Roughly 760 words each, four to five minutes read aloud.

Because the two scripts share almost no vocabulary, each one's term list
acts as a **distractor list** for the other clip — the same oracle-versus-
distractor contrast Earnings-21 ships, for free.

## Recording

Read the text **exactly as written**. Do not paraphrase, reorder, or
ad-lib: the file is the reference transcript, so any deviation is scored
as an ASR error that the ASR did not make.

Numbers and dates are spelled out on purpose ("twenty twenty four", not
"2024"). Written and spoken forms have to match or biasing cannot fire on
them. Read the two acronyms, MCP and RAG, as you naturally would.

Normal pace, quiet room, one take per script. Restarts are fine if you
re-read the whole sentence; leave the audio uncut.

Save each recording next to its script, same basename:

```
read_aloud/ai_agents.txt          read_aloud/ai_agents.wav
read_aloud/distributed_systems.txt   read_aloud/distributed_systems.wav
```

Any format ffmpeg reads is fine; convert to what the model wants with:

```bash
ffmpeg -i ai_agents.m4a -ar 16000 -ac 1 read_aloud/ai_agents.wav
```

`read_aloud/` is then a valid clip directory for both `captionlm/eval.py`
and `scripts/sweep_cb_weight.py` with no further glue — `load_clips`
pairs `<name>.wav` with `<name>.txt` and picks up an optional
`<name>.terms.txt`.

## Running it

Extract a term list per document, then sweep:

```bash
PYTHONPATH=. venv/bin/python scripts/extract_terms.py \
  read_aloud/ai_agents.txt --mode combined --out /tmp/ai.txt
PYTHONPATH=. venv/bin/python scripts/extract_terms.py \
  read_aloud/distributed_systems.txt --mode combined --out /tmp/ds.txt

PYTHONPATH=. venv/bin/python scripts/sweep_cb_weight.py read_aloud \
  --condition ai=/tmp/ai.txt --condition ds=/tmp/ds.txt \
  --score-terms /tmp/ai.txt --weights 0.25,0.5,1.0,2.0
```

`--mode` selects the extraction hypothesis: `ner` (proper nouns and rare
jargon), `terminology` (pyate's statistical phrases, filtered), or
`combined`.
