# Drop-off

Put a recording here with the document it is about. Get back an `.srt`
in which the domain terms are spelled correctly.

## How

Two files, **same basename**:

```
dropoff/standup.m4a      <- the recording
dropoff/standup.pdf      <- the document it is about
```

Then, from the repo root:

```bash
PYTHONPATH=. venv/bin/python scripts/caption_dropoff.py
```

You get:

```
dropoff/standup.srt          <- the captions
dropoff/standup.terms.txt    <- the terms it biased toward, so you can see what it decided
```

Audio: `.mp3 .mp4 .m4a .wav .mov .aac .flac .ogg .webm .mkv` (anything
ffmpeg reads; non-wav is converted automatically).
Documents: `.txt .md .pdf .docx .doc .csv .rtf .html`.

Drop in several pairs at once — it processes all of them and loads the
model only once.

An audio file with no matching document still gets transcribed, unbiased.
That is the honest baseline, and it is the fastest way to see what the
biasing is actually buying you: run it once without the document, once
with, and diff the two `.srt` files.

## Options

```bash
# the larger model: better on ordinary words, 4 GB download, ~2x slower
PYTHONPATH=. venv/bin/python scripts/caption_dropoff.py \
  --model mlx-community/parakeet-tdt_ctc-1.1b

# ask a second, independent model and let it correct the spans the two
# disagree on. The single biggest quality option here: measured at 1.1b,
# WER 0.0712 -> 0.0560 on read-aloud and 0.1918 -> 0.1837 on real
# earnings calls, with domain-term recall going UP on both. 1.6 GB
# download, roughly a quarter again of the runtime.
--second-opinion

# how the terms are chosen
--mode combined      # names + rare nouns + statistical phrases (default)
--mode ner           # names and rare nouns only
--mode terminology   # pyate's statistical phrases only

--top-n 50           # cap the term list (default: no cap)
--cb-weight 3.0      # biasing strength (default 3.0)
```

`--cb-weight` was swept on the read-aloud eval set. The curve is an
inverted U: nothing happens below 0.5, the gain peaks at 3.0–4.0, and it
collapses past 6.0 (WER 0.20 at 8.0, 0.47 at 12.0). Leave it at 3.0
unless you are experimenting.

## What to expect

Measured on two recorded technical talks where the script was also the
reference, biasing moved WER from 0.1160 to 0.0969 on the small model and
0.1009 to 0.0778 on the large one, and term F-score from 0.86 to 0.95. It
reliably fixes names and jargon — `Anthropic` heard as "entropic", `Paxos`
as "paxels", `quorum` as "core", `Merkle` as "mercult".

It will **not** fix:

- **Homophones.** `Groq` and `Grok` are acoustically identical; a term
  list has no way to choose between them. Same for `write`/`right`.
- **Grammar and function words.** Dropped articles, wrong verb tense.
  About two thirds of the remaining errors are these, and they are the
  acoustic model's job, not the term list's.
- **Stutters and restarts.** If the speaker says a phrase twice, you get
  it twice — that is a faithful transcription, not a bug.

One caveat on those numbers: in that test the document *was* the
transcript, which is the best case. Your document is merely *related* to
your audio, so expect less. On real earnings calls, only about 11% of the
fixable errors appeared anywhere in the company's own document — the hard
words are usually people's names, which press releases do not contain.
If you have a list of who is speaking, adding their names to
`<basename>.terms.txt` by hand and re-running is the single highest-value
thing you can do.

Files you drop here are git-ignored except this README.
