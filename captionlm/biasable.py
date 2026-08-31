"""Turn a document into a term list worth *biasing*, which is not the same
thing as a term list that describes the document.

pyate does automatic terminology extraction: statistically salient n-grams
for a domain. Contextual biasing needs something different -- words the
ASR will get WRONG. Measured on a real SEC earnings release, pyate's top
50 were 38% spoken-in-the-call and had ZERO overlap with Earnings-21's own
oracle bias list. It returned "+", "long", "line", "lease", and
"line results in the second quarter"; the oracle list for that same call
holds "MONRO FORWARD" and "MONRO UNIVERSITY".

Two failures, both fixed here:

1. Common words. The model already transcribes "revenue" and "shares"
   correctly, so biasing them has no recall upside and real
   false-accept downside. Single-word terms must therefore be a proper
   noun, a named entity, or an acronym to survive.

2. Fragments. "line results in the second quarter" is a slice of a
   sentence, not a term -- and because cb_weight is applied once per
   token in the spotter, a long entry accumulates the largest bonus and
   defeats the false-accept guard hardest. Terms carrying function words
   or running past MAX_TERM_WORDS are dropped.

Commonness is measured, not guessed. spaCy's en_core_web_sm ships no
frequency table (`rank` is unset and `is_oov` is True even for "the")
and the model's 1024-piece sentencepiece vocabulary cannot separate
3-piece "revenue" from 3-piece "Alcoa", but pyate -- already a
dependency -- bundles a 3000-document general-English corpus for its own
statistical scoring. Counting it gives 1.56M tokens, enough to put
"line" at 271 per million and "quorum" at 3.2.

That measurement also collapses two rules into one. Every term worth
biasing is rare in general English, whether it is a proper noun
(Anthropic 0.0, Alcoa 1.9, Raft 5.1) or ordinary-looking domain jargon
(linearizability 0.0, idempotent 0.6, tombstone 1.9). Every term worth
rejecting is common (line 271, revenue 39, node 11.5). So a single
rarity gate replaces the part-of-speech heuristic that used to guard
single words -- and fixes what that heuristic could not: it kept "node"
(spaCy tags it PROPN in a carrier sentence) and dropped "quorum".
"""
import collections
import functools
import re

import spacy

MAX_TERM_WORDS = 4
MIN_TERM_CHARS = 3

# Occurrences per million in general English, above which a word is common
# enough that the model already transcribes it correctly. Calibrated on the
# corpus below: it must reject "node" (11.5) while keeping "raft" (5.1).
MAX_COMMON_PER_MILLION = 10.0

# Entity types that name things a speaker says aloud and an ASR is likely
# to miss. Dates, times, money, percentages and ordinals are deliberately
# excluded: they are spoken in forms that never match their written shape
# ("Q3 2020" is said "third quarter twenty twenty"), so biasing the
# written form cannot fire. See the alternate-spelling approach in
# arXiv:2209.01250 for what would be needed to use them.
ENTITY_LABELS = {"ORG", "PERSON", "PRODUCT", "GPE", "FAC", "NORP", "LAW", "EVENT", "WORK_OF_ART"}

_nlp = None


def _pipeline():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


@functools.lru_cache(maxsize=1)
def _general_frequencies() -> tuple[collections.Counter, int]:
    """Word counts over pyate's bundled general-English corpus.

    pyate loads only 300 of the 3000 documents by default, which is too
    few to be usable: at that size "quarter" and "revenue" occur three and
    five times, indistinguishable from "raft" at one. The full corpus is
    1.56M tokens and takes ~0.3s to count, once per process.
    """
    from pyate.term_extraction import TermExtraction

    counts: collections.Counter = collections.Counter()
    for doc in TermExtraction.get_general_domain(size=10**9):
        counts.update(w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]*", str(doc)))
    return counts, sum(counts.values())


def is_rare(word: str) -> bool:
    """Is this word uncommon enough in general English to be worth biasing?"""
    counts, total = _general_frequencies()
    return 1e6 * counts[word.lower()] / total <= MAX_COMMON_PER_MILLION


def _is_acronym(word: str) -> bool:
    return len(word) >= 2 and word.isupper() and word.isalpha()


def is_biasable(term: str, nlp=None) -> bool:
    """Would biasing this term plausibly help more than it hurts?"""
    term = term.strip()
    words = term.split()
    if not words or len(words) > MAX_TERM_WORDS:
        return False
    if len(term) < MIN_TERM_CHARS or not re.search(r"[A-Za-z]", term):
        return False

    nlp = nlp or _pipeline()
    stops = nlp.Defaults.stop_words

    # A function word anywhere marks a sentence fragment, not a term:
    # "line results in the second quarter", "share of costs".
    if any(w.lower() in stops for w in words):
        return False

    if len(words) == 1:
        word = words[0]
        # A capitalized single word is a proper noun by orthography, which
        # is more reliable than either tagging it or measuring it: "Apple"
        # is a company however common "apple" is. Callers that harvest from
        # a document must therefore not hand us a word whose only capital
        # is sentence-initial.
        return _is_acronym(word) or word[0].isupper() or is_rare(word)

    return True


def _candidates(doc):
    """Surface forms in one parsed chunk that might be worth biasing.

    Three harvests, because no one of them is sufficient:

    1. Named entities. Reliable for names the NER model has seen.
    2. Capitalized runs. Necessary because en_core_web_sm assigns NO
       entity label at all to domain-new proper nouns -- Anthropic,
       Ollama, Raft, Perplexity and Alibaba are all invisible to harvest
       1 -- while orthography still marks them. A sentence-initial
       capital is only orthographic, so it must clear the rarity bar to
       count; that keeps out "Begin" and "Below" while still admitting
       "Ollama", which happens to appear exactly once, at the start of
       its only sentence.
    3. Rare lowercase words: quorum, tombstone, sharding.
    4. Bigrams anchored on a rare word, whose other half is nominal or
       adjectival. The multi-word jargon an ASR mishears -- bloom filter,
       hinted handoff, fencing token -- is lowercase, so harvest 2 cannot
       see it, and pyate ranks most of it below its whole output. Anchor
       on the rare word rather than parsing: en_core_web_sm mis-chunks
       unfamiliar noun-noun compounds, returning "a fencing" without
       "token" and "brain" without "split". Requiring the neighbour to be
       nominal is what keeps "tooling actually" out; a noun_chunks
       harvest was tried alongside this and removed, because it found
       nothing the bigrams miss and added fragments of its own.

    Two known gaps, both needing a domain lexicon rather than a wider
    rule. A compound of two common words ("split brain") clears no rarity
    bar. A compound whose other half tags as a verb ("write skew", where
    "write" is a VERB in "permits write skew") is excluded by the rule
    above; admitting verbs costs more fragments than it recovers terms.
    Compounds the tagger reads as legitimate noun pairs still slip
    through -- "Alibaba ships", "dense open" -- and are left to the
    false-accept measurement rather than guessed at here.
    """
    tokens = list(doc)
    for ent in doc.ents:
        if ent.label_ in ENTITY_LABELS:
            yield ent.text

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if (
            token.is_alpha
            and token.text[0].isupper()
            and (not token.is_sent_start or is_rare(token.text))
        ):
            end = i
            while end + 1 < len(tokens) and tokens[end + 1].is_alpha and tokens[end + 1].text[0].isupper():
                end += 1
            yield " ".join(t.text for t in tokens[i : end + 1])
            i = end + 1
            continue
        i += 1

    for token in tokens:
        if token.is_alpha and token.text.islower() and is_rare(token.text):
            yield token.text

    for i, token in enumerate(tokens):
        if not (token.is_alpha and is_rare(token.text)):
            continue
        for neighbour in (i - 1, i + 1):
            if 0 <= neighbour < len(tokens):
                other = tokens[neighbour]
                # Only a nominal or adjectival neighbour makes a term. Without
                # this the harvest emits sentence fragments -- "Alibaba ships",
                # "tooling actually", "window fills" -- which score well on a
                # read-aloud set purely because the document IS the transcript,
                # and are pure noise on any real document.
                if other.is_alpha and not other.is_stop and other.pos_ in ("NOUN", "PROPN", "ADJ"):
                    pair = sorted((i, neighbour))
                    yield f"{tokens[pair[0]].text} {tokens[pair[1]].text}"


def extract_entities(text: str, top_n: int | None = None) -> list[str]:
    """Terms worth biasing from the document, most frequent first.

    This is the class of term Earnings-21's oracle bias list is made of --
    ALCOA, AIRBUS, ALAN SCHNITZER -- and the class pyate never returns.
    """
    nlp = _pipeline()
    counts: dict[str, int] = {}
    surface: dict[str, str] = {}
    for doc in nlp.pipe(_chunks(text), batch_size=8):
        for candidate in _candidates(doc):
            name = " ".join(candidate.split())
            if not is_biasable(name, nlp):
                continue
            key = name.lower()
            counts[key] = counts.get(key, 0) + 1
            surface.setdefault(key, name)

    ranked = sorted(counts, key=lambda k: (-counts[k], k))
    terms = [surface[k] for k in ranked]
    return terms[:top_n] if top_n else terms


def _chunks(text: str, size: int = 90_000):
    """spaCy's parser caps input length; split on whitespace near the cap."""
    for i in range(0, len(text), size):
        yield text[i : i + size]


def filter_biasable(terms: list[str]) -> list[str]:
    nlp = _pipeline()
    return [t for t in terms if is_biasable(t, nlp)]


def merge_term_lists(*lists: list[str]) -> list[str]:
    """Union preserving first-seen order, deduplicated case-insensitively."""
    seen: set[str] = set()
    merged: list[str] = []
    for terms in lists:
        for term in terms:
            key = term.lower()
            if key not in seen:
                seen.add(key)
                merged.append(term)
    return merged
