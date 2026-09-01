# A second opinion on the same audio

The error budget said document-conditioned rescoring was the only lever big
enough to reach ~5% WER, and that everything else combined was worth about
0.013. Rescoring was tried, four ways, and none of them worked. What did
work was not rescoring at all: a second acoustic model, and three rules
about when it is allowed to overrule the first.

**Read-aloud, `parakeet-tdt_ctc-1.1b`, cb_weight 3.0, each clip biased with
its own extracted term list:**

| | errors / 1517 | WER |
|---|---|---|
| biased parakeet (shipped before this) | 108 | 0.0712 |
| **+ second opinion** | **85** | **0.0560** |

That is −0.0152, against −0.0073 for every previous change on this project
put together. It clears the budget's "everything except rescoring → 0.065"
line and lands most of the way to the 0.05 target.

## Why rescoring failed

Four attempts, all with Qwen3-1.7B-4bit or Llama-3.2-3B-Instruct-4bit, both
comfortably inside the 2 GB budget. All measured on the same set.

| attempt | errors | what went wrong |
|---|---|---|
| prompt the LLM to fix each sentence | 120 | it paraphrases: `Mistral`→`Nvidia`, `they`→`agents`, `disagree`→`agree` |
| same, but only accept edits at non-words | 104 | safe, and it only ever reaches the ~8 garbled words |
| detect non-words, let the LLM pick from term-list candidates | 104 | it ignores acoustics: `PEXLs`→`permits`, `Bobpeka`→`problem` |
| score inflection variants by LM log-prob, take the best | 118 | fluency is not what was said |

The common failure is that a language model asked to improve a transcript
has no way to know what the speaker actually said. Every proposal it makes
is invented, and the guard rails that stop it inventing also stop it
helping. The one place it was reliably right — replacing a word that is not
an English word — is reachable without it: plain string similarity against
the term list scores 101 errors, three better than any LLM variant, in
milliseconds and with nothing to download.

## What the second model brings that a language model cannot

Whisper large-v3-turbo is *worse* than biased parakeet on this set — 197
errors against 108, 0.1299 against 0.0712 — so it cannot simply replace it.
But its errors are somewhere else. An oracle allowed to pick the better arm
at every disagreement scores 58 errors, 0.0382. Fifty error words sit in
that gap.

Unlike an LLM's proposals, Whisper's have heard the audio. That is the whole
difference.

### Three rules, each measured

Taking Whisper at every disagreement is much worse than not doing it at all.
Adding the rules one at a time:

| | errors | WER |
|---|---|---|
| biased parakeet alone | 108 | 0.0712 |
| take Whisper at every disagreement | 195 | 0.1285 |
| + reject malformed Whisper spans | 148 | 0.0976 |
| + never drop a spotted term | 89 | 0.0587 |
| + reject pure insertions | 85 | 0.0560 |

That column is cumulative and so depends on the order they are added in. The
order-free question is what each rule is worth in the shipped configuration,
by removing just that one:

| rule removed from the shipped set | errors | costs |
|---|---|---|
| — (shipped) | 85 | |
| reject malformed spans | 100 | 15 |
| never drop a spotted term | 106 | 21 |
| reject pure insertions | 89 | 4 |

**Malformed spans.** Whisper's failure mode here is formatting, not hearing:
`writeahead log`, `logstructured mergetree`, `finetuning`, `poorlytuned`,
`singleobject`, and `10` for "ten". Each is a WER error against a verbatim
reference and none is a better guess than what parakeet produced. A span is
refused if any word in it holds a digit, or is absent from both general
English and the speaker's term list. Worth 15 error words.

The frequency table this uses is pyate's bundled general-English corpus,
already counted and cached for `biasable`. The system dictionary at
`/usr/share/dict/words` was tried alongside it and changed the result by
exactly zero words, so it is not used — one less thing to depend on, and no
question about which platforms ship it.

**Never drop a spotted term.** Whisper is not biased and cannot know the
speaker's vocabulary, so wherever it would spend a term the spotter found,
it is wrong by construction. Worth 21 error words — the largest of the three,
and the one that makes fusion compatible with biasing rather than a
replacement for it.

**Pure insertions.** Words only Whisper heard are usually its own repetition
across a segment boundary. Deletions are kept, though: where Whisper heard
*fewer* words, both models heard the stretch and only one transcribed the
reader's false start. Worth 4 error words.

Relaxing this to allow single function words — parakeet's known deletion
failure mode, and 19 words of the budget — was tried and is worse: 87 errors
against 85. Six of the nine insertions Whisper offers are `the`, `it` or
`that`, and they land in the wrong places often enough to lose more than they
win. There is no narrower version of this rule that pays.

### What it actually fixes

The spans it wins are the ones no term list could reach:

- `bar rails catches` → `guardrails catch`. Merge collateral: the spotter
  replaced a span it only partly accounted for. Three error words in one.
- `different machinery` → `vector clock`, `amplification` → `sorted run`,
  `core` → `quorum`. Acoustically distant substitutions, the largest bucket
  in the budget and the one it had no plan for.
- `agents holding` → `agent tooling`, `protection` → `production`,
  `about` → `without`. Ordinary content words.
- `it might take` → nothing. A reader restart the n-gram disfluency pass
  could not see, because the reader restarted with different words.

What it does **not** fix is the one case the budget singled out. `Groq` is
still `Grok`: Whisper hears `Grok` there too, so there is no disagreement to
arbitrate. The budget was right that only the surrounding sentence separates
them, and wrong that this made it reachable — a second acoustic opinion is
not a semantic one, and the four language-model passes that could have been
semantic all cost more words than they saved.

### It is not tuned to one operating point

The same Whisper transcript, fused against the biased hypothesis at each
`cb_weight` the sweep stored:

| cb_weight | biased | fused |
|---|---|---|
| 2.0 | 116 errors, 0.0765 | 93 errors, 0.0613 |
| 3.0 | 108 errors, 0.0712 | **85 errors, 0.0560** |
| 4.0 | 111 errors, 0.0732 | 86 errors, 0.0567 |

Twenty-three, twenty-three and twenty-five error words. The gain does not
come from having landed on a lucky weight, and 3.0 is still the floor of the
curve after fusion as it was before.

### Biasing the second model too — declined

Whisper takes an `initial_prompt`, and feeding it the same term list looks
like free extra recall on the arm that has none. It is the opposite: Whisper
alone goes from 197 errors to 254, and fusing against it from 85 to 97. The
prompt drags its style and it starts producing the list rather than the
audio. The second opinion is only useful while it stays an independent one.

## An LLM arbiter, measured and declined

The obvious next move is to arbitrate the remaining disagreements with a
language model rather than a flat rule: score both arms under Qwen3-1.7B-4bit
in context and take Whisper only when it wins by a margin. It works — 84
errors at a margin of 2 nats, with 26 swaps where the rules make 48.

Eighty-four against eighty-five. One word, for a second 1 GB model to
download and a forward pass per disagreement. It is not shipped. The rules
are cheaper and, on two clips, indistinguishable.

The same goes for stacking the non-word repair on top of fusion: 83 errors,
two words better, but it changes 20 words on Earnings-21 for +0.0001 WER
there. Two mechanisms doing one job, for two words. Not shipped.

## The thing this measurement found by accident

**44 of parakeet's 108 error words are made identically by Whisper** — 38
distinct substitutions, produced by two independent architectures on the same
audio. A handful may be genuinely hard audio that both models lose the same
way, but agreement at this rate is not chance: most of these are places where
the reader deviated from the script.

    networks → network      drop → drops        packets → packages
    produce → produces      live → lives        exists → exist
    stronger → strong       shorter → short     objects → object
    stop → stopped          subtle → slow       engine → machine
    dwarfs → goes           this → it           that → the

The read-aloud README requires the reader to read the text exactly as
written, because the file is simultaneously the source document and the
reference transcript. They did not, in roughly forty places, and every one
has been scored as an ASR error in every number this project has recorded.

Two consequences.

First, **the error budget's inflection bucket is not an ASR failure.**
`protocol`→`protocols`, `objects`→`object`, `exists`→`exist` and most of
their neighbours are correct transcriptions of what was said. That is why
LM-scored inflection correction made things worse: it was changing right
answers to match a script the reader had not followed.

Second, **the 0.05 target is closer to the floor than it looked.** If those
44 words are reference mismatch rather than recognition error, the reachable
floor on this set is 0.029 WER, and the 76-error target sits about 32 errors
above it rather than the full 108. This does not make the target easier — it
makes the remaining pool smaller and harder than the budget assumed.

It also bounds what fusion could ever have been worth. A shared error
produces no disagreement, so fusion cannot see one. Of the original 108, only
64 were ever in reach; fusion took 23 of those, and the 85 still wrong are 44
it could never have seen plus 41 it saw and left.

Neither the audio nor the reference was changed to reflect this. Re-cutting
the reference to what the reader said would invalidate every stored result
in `eval_results/`, and the honest version of the caveat belongs in the
numbers' interpretation, not in a silent edit to the ground truth.

## Cost and how to use it

`mlx-community/whisper-large-v3-turbo` is 1.61 GB and adds roughly a quarter
again to transcription time. It is opt-in, following the same principle as
`--model`: a download this size is the user's disk and the user's choice.

```bash
PYTHONPATH=. venv/bin/python -m captionlm.cli talk.wav \
  --terms terms.txt --model mlx-community/parakeet-tdt_ctc-1.1b --second-opinion
```

`captionlm/eval.py` takes the same flag, so the number above is reproducible
through the scored path rather than only over stored predictions.

## It does not cost term recall, and it holds on real audio

The headline pair for this project is recall against injection, not WER, and
a WER win bought by dropping domain terms would be a bad trade. It is not
that trade. Scored the same way `eval.py` does:

| | WER | precision | recall | F | injected |
|---|---|---|---|---|---|
| **read-aloud** (1517 words) | | | | | |
| unbiased | 0.0962 | 0.9901 | 0.7909 | 0.8794 | — |
| biased | 0.0712 | 0.9770 | 0.9231 | 0.9493 | — |
| Whisper alone | 0.1299 | 0.9828 | 0.7890 | 0.8753 | — |
| **biased + second opinion** | **0.0560** | 0.9775 | **0.9428** | **0.9598** | — |
| **Earnings-21** (5 calls, 40003 words) | | | | | |
| unbiased | 0.1910 | 0.9093 | 0.9682 | 0.9378 | 6/781 |
| biased | 0.1918 | 0.9093 | 0.9682 | 0.9378 | 6/781 |
| Whisper alone | 0.1293 | 0.9635 | 0.9733 | 0.9684 | 2/781 |
| **biased + second opinion** | **0.1837** | 0.9100 | **0.9759** | **0.9418** | **4/781** |

Recall goes **up** on both sets, and on Earnings-21 the injection count goes
down, from 6 to 4. That is the "never drop a spotted term" rule doing what it
was written for: the only spans Whisper is allowed to take are ones where
biasing had nothing at stake, so its general-speech strength is added rather
than traded against.

The Earnings-21 arm is a real generalisation test — spontaneous, multi-speaker,
verbatim references, a generic 193-term list rather than a per-clip extraction,
and audio nothing here was tuned on. It gains 0.0081. Half of read-aloud's
0.0152, in the same direction, and with the term metrics moving the right way
on both.

## The finding that should make somebody uncomfortable

**On Earnings-21, Whisper alone beats biased parakeet by a wide margin:
0.1293 against 0.1918.** On read-aloud the ranking is reversed and not close,
0.1299 against 0.0712. Same two models, opposite verdicts, decided entirely
by the material: parakeet 1.1b is excellent on clean scripted single-speaker
audio and much weaker on spontaneous conversational speech, which is what a
real user records.

Swapping the backbone was the obvious response, and it was measured: run the
same rules with Whisper as the backbone and parakeet as the second opinion,
with a listed term winning from whichever arm holds it.

| backbone | read-aloud | Earnings-21 |
|---|---|---|
| parakeet (shipped) | **85 errors, 0.0560** | **0.1837** |
| Whisper | 89 errors, 0.0587 | 0.1850 |
| Whisper alone, no fusion | 197 errors, 0.1299 | **0.1293** |

Worse both times, and the Earnings-21 row is the surprising one: on audio
where Whisper alone wins by 0.05, letting parakeet correct it gives most of
that back. Whichever model is behind, these rules protect the backbone, and
protecting the wrong backbone costs more than the fusion gains.

So the shipped direction is right and the question is not about fusion at
all. It is that **on spontaneous speech, plain Whisper beats this project's
entire biased pipeline — on WER (0.1293 against 0.1918), on term precision
(0.9635 against 0.9093), and on term recall (0.9733 against 0.9682)**, with
no term list and nothing to extract. Biasing earns its keep on the read-aloud
set, where recall is 0.7890 unbiased against 0.9231 biased, and that set is
clean scripted single-speaker audio read from the reference itself.

Nothing here settles it: two term lists, one of them generic, and nine calls.
But it is the measurement that should decide what gets built next, and it
points away from tuning the spotter.

## The caveat that still governs all of it

Unchanged from the budget: the read-aloud set is one speaker, clean audio,
scripted prose, and a document that IS the transcript. The Earnings-21 arm
above is five calls of nine, chosen only by which had finished transcribing,
and scored against a term list built for the whole corpus rather than for each
call; treat 0.0081 as a direction, not a decimal.
