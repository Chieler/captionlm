# captionlm

Captions that spell your domain terms correctly. Give it a recording and the
document it is about; get back an `.srt` in which `Paxos`, `linearizability`
and `Groq` are not `taxes`, `delarizability` and `grok`.

Apple silicon, MLX, entirely local. **Nothing is trained.** The term list is
compiled into a trie at call time, which is why the wait is transcription
only.

## Quickstart

See `SETUP.md` first — there are two `pyate` install landmines.

```bash
cp recording.m4a notes.pdf dropoff/            # same basename
venv/bin/python scripts/serve.py               # http://localhost:8756
```

or without the browser:

```bash
venv/bin/python scripts/caption_dropoff.py --second-opinion
```

## What it costs

Measured on the read-aloud set: 447 terms over 507 spoken occurrences,
681.5s of audio.

| model | second opinion | WER | term recall | speed | disk |
|---|---|---|---|---|---|
| `parakeet-tdt_ctc-110m` | no | 0.0850 | 0.9073 | 59x | 438 MB |
| `parakeet-tdt_ctc-110m` | yes | 0.0659 | 0.9250 | 12x | +1.6 GB |
| `parakeet-tdt_ctc-1.1b` | no | 0.0712 | 0.9231 | 30x | 4.0 GB |
| `parakeet-tdt_ctc-1.1b` | yes | 0.0560 | 0.9428 | 10x | +1.6 GB |

Those are clean single-speaker audio with a matching source document. On
spontaneous speech (Earnings-21) the same pipeline sits at 0.1858, and plain
Whisper beats it on WER while losing on term recall. `docs/results/` has the
measurements and the things that did not work.

## How it works

1. `biasable.py` pulls domain terms out of the document.
2. `terms.py` compiles them into a context graph (trie) at call time.
3. `biased_model.py` runs parakeet TDT-CTC; a spotter reads the CTC head for
   term hits and `merge.py` folds them into the TDT hypothesis by frame
   overlap.
4. `fusion.py` optionally transcribes again with Whisper and accepts its
   corrections where three guards allow it — never dropping a spotted term,
   never a pure insertion, only well-formed spans.
5. `cli.py` writes the SRT, dropping the speaker's false starts.

CTC-WS is the reason the architecture is fixed: the spotter needs a CTC head
over the same sentencepiece vocabulary, so the model family cannot change.

## Layout

```
captionlm/   the package -- everything above happens here
scripts/     entry points, plus dashboard.html for scripts/serve.py
tests/
docs/        results/ is the measurement record; superpowers/ is plans and specs
dropoff/     drop a recording and its document here
data/        eval sets, results, corpora -- gitignored except read_aloud/
```
# captionlm
# captionlm
# captionlm
