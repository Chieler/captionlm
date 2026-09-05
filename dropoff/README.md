# Drop-off

Put a recording here with the document it is about. Get back an `.srt` and `.txt`
in which the domain terms are spelled correctly.

## How

Two files, **same basename**:

```
dropoff/standup.m4a      <- the recording
dropoff/standup.pdf      <- the document it is about
```

Several documents can bias one recording. A document belongs to a
recording when its basename is the recording's, or starts with it followed
by `-`, `_`, `.` or a space:

```
dropoff/standup.m4a
dropoff/standup.pdf
dropoff/standup-notes.txt
dropoff/standup_slides.md
```

All three are read and their terms pooled. A document that could match two
recordings goes to the longer name, so `standup-part2.txt` biases
`standup-part2.m4a` rather than `standup.m4a`.

A document that names **no** recording is shared -- it biases every recording
in the drop-off:

```
dropoff/standup.m4a
dropoff/retro.m4a
dropoff/glossary.pdf     <- biases both
dropoff/standup.txt      <- biases standup.m4a alone
```

Real documents are called `glossary.pdf`, not `standup.pdf`, and requiring
the rename is a contract nobody keeps. Naming a document after a recording is
what scopes it to that recording alone.

Then, from the repo root, either the command line:

```bash
venv/bin/python scripts/caption_dropoff.py
```

or the browser, which does the same work with the two quality choices
visible instead of buried in flags:

```bash
venv/bin/python scripts/serve.py    # http://localhost:8756
```

The UI drops files into this directory, shows the transcript arriving as it
is produced, shows real transcription progress,
and marks the spans the second model corrected. With `--second-opinion` on,
the transcript is held back until the second model has read the audio --
those cues would otherwise be rewritten under you. The × next to a recording
or a stray file removes it from this directory, along with anything a run
wrote for it. It runs one job at a time and binds loopback only.

Remove a recording outside the UI and its `.srt` and `.terms.txt` are listed
as strays, to be removed with the same ×; they are never deleted for you,
because the `.srt` is the thing you came for. The `.converted.wav` beside it
is swept on sight -- it is a decoding cache the size of the recording, and
it is rebuilt on demand.

You get:

```
dropoff/standup.srt          <- the captions
dropoff/standup.transcript.txt <- the plain-text transcript
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
venv/bin/python scripts/caption_dropoff.py \
  --model mlx-community/parakeet-tdt_ctc-1.1b

# ask a second, independent model and let it correct the spans the two
# disagree on. The single biggest quality option here: measured at 1.1b,
# WER 0.0712 -> 0.0560 on read-aloud and 0.1929 -> 0.1858 on real
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
