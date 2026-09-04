"""Extract domain-specific terms from document text.

Uses pyate's term_extractor (Sclano & Velardi 2007), which scores POS-
tagged candidate phrases by frequency in the given text relative to
pyate's bundled general-English reference corpus — this is what makes it
domain-specific extraction rather than generic keyword extraction. MIT
license, no GPU, spaCy en_core_web_sm only.
"""
import numpy as np
from pyate import TermExtraction, term_extractor

from captionlm.config import TERM_EXTRACTION_TOP_N

# pyate's default uint16 counters overflow on long source documents such as
# SEC exhibits. term_extractor creates fresh TermExtraction instances
# internally, so configure its process-wide default at the integration
# boundary rather than trying to pass a dtype through its public wrapper.
TermExtraction.configure({"dtype": np.uint32})


def extract_terms(text: str, top_n: int = TERM_EXTRACTION_TOP_N) -> list[str]:
    scores = term_extractor(text)
    if scores.empty:
        return []

    ranked = scores.sort_values(ascending=False)
    seen: set[str] = set()
    terms: list[str] = []
    for term in ranked.index:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
        if len(terms) >= top_n:
            break
    return terms


def write_term_list(terms: list[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for term in terms:
            f.write(term + "\n")
