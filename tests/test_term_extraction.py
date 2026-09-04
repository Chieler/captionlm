# tests/test_term_extraction.py
from captionlm.term_extraction import extract_terms, write_term_list
from pyate.term_extraction import TermExtraction
import numpy as np

JARGON_DOC = """
Our platform runs on a kubernetes cluster deployed across three regions.
The kubernetes cluster autoscaler adjusts node pools based on demand.
We migrated the kubernetes cluster configuration to a new Helm chart last
quarter, and the on-call team monitors kubernetes cluster health daily.
The weather was nice today. I went for a walk and had a coffee. It was a
pretty ordinary afternoon with nothing much to report.
"""


def test_extract_terms_finds_repeated_jargon():
    terms = extract_terms(JARGON_DOC, top_n=10)
    assert any("kubernetes" in term.lower() for term in terms)


def test_extract_terms_deduplicates_case_insensitively():
    terms = extract_terms(JARGON_DOC, top_n=10)
    lowered = [t.lower() for t in terms]
    assert len(lowered) == len(set(lowered))


def test_extract_terms_uses_counter_wide_enough_for_long_documents():
    """pyate creates fresh internal counters while extracting; uint16
    overflows on long SEC exhibits before CaptionLM can filter their terms.
    """
    assert TermExtraction("ordinary document text").dtype == np.uint32


def test_write_term_list(tmp_path):
    path = tmp_path / "terms.txt"
    write_term_list(["kubernetes cluster", "Helm chart"], str(path))
    content = path.read_text(encoding="utf-8")
    assert content == "kubernetes cluster\nHelm chart\n"
