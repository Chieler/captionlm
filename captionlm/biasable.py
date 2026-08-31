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

Neither spaCy's en_core_web_sm (no frequency table: `rank` is unset and
`is_oov` is True even for "long") nor the model's 1024-piece
sentencepiece vocabulary (cannot separate 3-piece "revenue" from 3-piece
"Alcoa") gives a usable word-frequency signal, so commonness is decided
structurally instead.
"""
import re

import spacy

MAX_TERM_WORDS = 4
MIN_TERM_CHARS = 3

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
        if _is_acronym(word):
            return True
        # Single common nouns ("line", "revenue", "shares") are words the
        # model already gets right, so they carry risk without upside.
        #
        # Tag inside a carrier sentence, not bare: spaCy tags an isolated
        # "Alcoa" as NOUN and "Monro" as ADJ, but gets both right as PROPN
        # once they sit in ordinary syntax.
        doc = nlp(f"We discussed {word} in detail.")
        token = doc[2]
        return token.pos_ == "PROPN" or token.ent_type_ in ENTITY_LABELS

    return True


def extract_entities(text: str, top_n: int | None = None) -> list[str]:
    """Named entities from the document, most frequent first.

    This is the class of term Earnings-21's oracle bias list is made of --
    ALCOA, AIRBUS, ALAN SCHNITZER -- and the class pyate never returns.
    """
    nlp = _pipeline()
    counts: dict[str, int] = {}
    surface: dict[str, str] = {}
    for doc in nlp.pipe(_chunks(text), batch_size=8):
        for ent in doc.ents:
            if ent.label_ not in ENTITY_LABELS:
                continue
            name = " ".join(ent.text.split())
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
