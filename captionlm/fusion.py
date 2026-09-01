"""A second opinion on the same audio, from a second acoustic model.

The residual after biasing is mostly not domain vocabulary. Of the 108 error
words left on the read-aloud set at cb_weight 3.0, only about half touch a
listed term; the rest are ordinary content words, function words and merge
collateral that no term list can reach. Those are exactly the words a
general-purpose ASR gets right, so the cheapest new evidence available is a
second model's opinion of the same audio.

Whisper large-v3-turbo is worse than biased parakeet overall -- 0.1299
against 0.0712 -- so it cannot simply replace it. But the two disagree in
different places: an oracle allowed to pick the better arm at every
disagreement scores 0.0382. Fifty error words sit in that gap, which is more
than every other lever in the error budget put together.

What captures a large part of it is not a clever arbiter but three flat
rules about when Whisper is *not* allowed to win (read-aloud, 1517 reference
words):

    biased parakeet alone                 108 errors, 0.0712
    take Whisper at every disagreement    195 errors, 0.1285
    all three rules                        85 errors, 0.0560

Each rule, priced by removing just that one from the shipped set: malformed
spans 15 error words, spotted terms 21, pure insertions 4.

Not every error is reachable this way. Where both models make the *same*
mistake there is no disagreement to arbitrate, and that is 44 of the 108 --
mostly places where the reader departed from the script. Only 64 were ever
in reach; these rules take 23 of them.

Parakeet stays the backbone and keeps its timings; Whisper only ever
replaces a span parakeet already produced.

An LLM arbiter was tried in place of the last two rules -- score both arms
under Qwen3-1.7B-4bit, take Whisper only when it wins by a margin. It landed
at 84 errors against these 85, one word, for a second model to download and
a second forward pass per disagreement. Not worth it; the rules are the
result.
"""
import re
import string

import kaldialign
from parakeet_mlx.alignment import AlignedToken

from captionlm.biasable import _general_frequencies
from captionlm.merge import _group_words

WHISPER_MODEL_ID = "mlx-community/whisper-large-v3-turbo"

_WORD_CHARS = re.compile(r"[a-z0-9']+")


def second_opinion(audio_path: str, model_id: str = WHISPER_MODEL_ID) -> str:
    """Transcribe the same audio with an independent model."""
    import mlx_whisper

    return mlx_whisper.transcribe(
        audio_path, path_or_hf_repo=model_id, language="en"
    )["text"].strip()


def _key(word: str) -> str:
    # Token text carries its own leading space (that is how parakeet-mlx marks
    # a word boundary), so whitespace has to go before punctuation does.
    return word.strip().strip(string.punctuation).lower()


def _term_vocabulary(terms: list[str]) -> set[str]:
    return {w for t in terms for w in _WORD_CHARS.findall(t.lower())}


def _is_word(word: str, vocabulary: set[str]) -> bool:
    """Is this a word someone could have said, rather than a transcription
    artifact?

    Whisper's failure mode here is formatting, not hearing: it writes
    "writeahead log", "logstructured mergetree", "finetuning" and "10" where
    the reference has two words, or a spelled-out number. Every one of those
    is a WER error against a verbatim reference and none of them is a better
    guess than what parakeet already produced, so the whole span is refused.

    The test is pyate's bundled general-English corpus -- already counted for
    `biasable`, already cached -- plus the speaker's own term list. The system
    dictionary at /usr/share/dict/words was tried alongside it and changed
    nothing on the read-aloud set: every artifact it rejects, the frequency
    table rejects too.
    """
    if len(word) < 4 or word in vocabulary:
        return True
    counts, _ = _general_frequencies()
    if counts[word] > 0:
        return True
    for suffix in ("s", "es", "ed", "ing"):
        stem = word[: -len(suffix)]
        if word.endswith(suffix) and len(stem) >= 4 and (stem in vocabulary or counts[stem] > 0):
            return True
    return False


def _disagreements(primary: list[str], second: list[str]) -> list[tuple[int, int, int, int]]:
    """Maximal spans where the two transcripts differ, as index ranges into
    each: (primary_start, primary_end, second_start, second_end)."""
    out: list[tuple[int, int, int, int]] = []
    i = j = 0
    start: tuple[int, int] | None = None
    for a, b in kaldialign.align(primary, second, "*"):
        if a == b:
            if start:
                out.append((start[0], i, start[1], j))
                start = None
        elif start is None:
            start = (i, j)
        if a != "*":
            i += 1
        if b != "*":
            j += 1
    if start:
        out.append((start[0], len(primary), start[1], len(second)))
    return out


def accepted_replacements(
    primary: list[str], second: list[str], terms: list[str]
) -> dict[int, tuple[int, list[str]]]:
    """Which disagreement spans Whisper is allowed to win, keyed by the
    primary index the replacement starts at -> (exclusive end, new words).

    Words are compared lowercased and unpunctuated; the returned words are
    Whisper's, in that same reduced form.
    """
    vocabulary = _term_vocabulary(terms)
    out: dict[int, tuple[int, list[str]]] = {}
    for start, end, s_start, s_end in _disagreements(primary, second):
        replacement = second[s_start:s_end]
        if start == end:
            # A pure insertion: Whisper heard words where parakeet heard
            # none. Parakeet is the backbone precisely because it is the
            # better transcript; words only it is missing are usually
            # Whisper's own repetition across a segment boundary.
            continue
        if any(any(c.isdigit() for c in w) for w in replacement):
            continue
        if any(not _is_word(w, vocabulary) for w in replacement):
            continue
        if any(w in vocabulary for w in primary[start:end]) and not any(
            w in vocabulary for w in replacement
        ):
            # Whisper is not biased and cannot know the speaker's vocabulary.
            # Where it would spend a term the spotter found, it is wrong by
            # construction -- the most valuable of the three rules, worth 21 words.
            continue
        out[start] = (end, replacement)
    return out


def fuse_text(primary: str, second: str, terms: list[str]) -> str:
    """Fuse two transcripts, keeping the primary's words and casing except
    where a disagreement span is handed to the second."""
    words = primary.split()
    keys = [_key(w) for w in words]
    replacements = accepted_replacements(keys, [_key(w) for w in second.split()], terms)

    out: list[str] = []
    i = 0
    while i < len(words):
        if i in replacements:
            end, new = replacements[i]
            out.extend(_restyle(new, words[end - 1], _starts_sentence(words, i)))
            i = end
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def _starts_sentence(words: list[str], i: int) -> bool:
    return i == 0 or words[i - 1].rstrip("\"'").endswith((".", "!", "?"))


def _restyle(new: list[str], last: str, at_sentence_start: bool) -> list[str]:
    """Give replacement words the trailing punctuation of the span they stand
    in for, so a fused sentence still reads as one.

    Casing is deliberately not inherited. The span being replaced is often a
    proper noun the second model did not hear -- "Paxos" for "quorum" -- and
    carrying its capital across would plant one mid-sentence. Only a span
    that opens a sentence gets capitalised.
    """
    if not new:
        return []
    styled = list(new)
    if at_sentence_start:
        styled[0] = styled[0][:1].upper() + styled[0][1:]
    styled[-1] += last[len(last.rstrip(string.punctuation)):]
    return styled


def fuse_tokens(
    tokens: list[AlignedToken], second: str, terms: list[str]
) -> list[AlignedToken]:
    """Same fusion, over aligned tokens, so captions keep their timings.

    A replaced span collapses to one token spanning the same time as the
    words it stands in for -- the same shape `merge.py` uses when a spotted
    keyword replaces a transducer span.
    """
    words = _group_words(tokens)
    keys = [_key("".join(t.text for t in w)) for w in words]
    replacements = accepted_replacements(keys, [_key(w) for w in second.split()], terms)

    out: list[AlignedToken] = []
    i = 0
    while i < len(words):
        if i in replacements:
            end, new = replacements[i]
            if new:
                first, last = words[i], words[end - 1]
                previous = "".join(t.text for t in words[i - 1]).strip() if i else ""
                text = " ".join(
                    _restyle(
                        new,
                        "".join(t.text for t in last),
                        not i or previous.rstrip("\"'").endswith((".", "!", "?")),
                    )
                )
                lead = " " if first[0].text.startswith(" ") else ""
                out.append(
                    AlignedToken(
                        id=-1,
                        text=lead + text,
                        start=first[0].start,
                        duration=last[-1].end - first[0].start,
                        confidence=1.0,
                    )
                )
            i = end
        else:
            out.extend(words[i])
            i += 1
    return out
