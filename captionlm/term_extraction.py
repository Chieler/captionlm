"""Extract domain-specific terms from document text.

Uses pyate's term_extractor (Sclano & Velardi 2007), which scores POS-
tagged candidate phrases by frequency in the given text relative to
pyate's bundled general-English reference corpus — this is what makes it
domain-specific extraction rather than generic keyword extraction. MIT
license, no GPU, spaCy en_core_web_sm only.
"""
from pyate import term_extractor


def extract_terms(text: str, top_n: int = 50) -> list[str]:
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
