"""Drop reader restarts from a finished transcript.

A restart is the speaker aborting a phrase and saying it again: "Groq built
custom silicon, Groq built custom silicon for low latency decoding". The
acoustic model transcribes it faithfully -- the unbiased baseline emits the
duplicate too, so this is not biasing damage -- but a caption that shows the
speaker's false starts is worse than one that does not, and the duplicated
words are pure insertions against any reference.

Detection uses the hypothesis alone. The eval-time way to find a restart is
to ask which inserted words also appear in the reference nearby, but there
is no reference at inference time, so that test cannot ship. What ships is
the property that makes a restart a restart: an n-gram immediately followed
by itself.
"""
import re

MAX_RESTART_WORDS = 6

# Below this length a repeat must be exact. A single mismatch inside a
# 2-gram leaves only one word actually matching, and ordinary English
# supplies those constantly -- "in the end the beginning" would collapse to
# "end the beginning".
MIN_FUZZY_WORDS = 3

# The mismatching pair in a fuzzy repeat has to look like one word twice,
# not two words. "grok"/"grock" share three characters; "enough"/"modes"
# share none, and without this guard that pair matched a window shifted one
# word off the real repeat in "the failure modes the failure modes",
# deleting "enough the failure" instead.
MIN_SHARED_PREFIX = 2


def _key(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


def _same_word_garbled(a: str, b: str) -> bool:
    shared = 0
    for x, y in zip(a, b):
        if x != y:
            break
        shared += 1
    return shared >= MIN_SHARED_PREFIX


def _is_repeat(keys: list[str], i: int, n: int, fuzzy: bool) -> bool:
    a, b = keys[i : i + n], keys[i + n : i + 2 * n]
    if not all(a) or not all(b):
        return False
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    if not diff:
        return True
    if not fuzzy or n < MIN_FUZZY_WORDS or len(diff) > 1:
        return False
    # One word may differ, because the word the speaker aborted on is
    # usually the hard one and comes back misrecognized rather than
    # identical: "grok built custom silicon grock built custom silicon".
    return _same_word_garbled(*diff[0])


def strip_restarts(text: str, max_n: int = MAX_RESTART_WORDS) -> str:
    """Remove the first copy of any immediately-repeated word sequence.

    Longest n first: "the failure modes the failure modes" must be caught as
    one 3-gram, not as three unrelated unigrams, which would leave the words
    interleaved. Exact repeats are taken before fuzzy ones for the same
    reason at the other end -- a fuzzy window sitting one word off a real
    repeat scores as well as the repeat itself, so the exact match has to be
    consumed while it is still there. The second copy is the one kept: it is
    what the speaker settled on, and it carries the sentence punctuation.
    """
    words = text.split()
    keys = [_key(w) for w in words]
    for fuzzy in (False, True):
        for n in range(max_n, 0, -1):
            i = 0
            while i + 2 * n <= len(words):
                if _is_repeat(keys, i, n, fuzzy):
                    del words[i : i + n]
                    del keys[i : i + n]
                else:
                    i += 1
    return " ".join(words)
